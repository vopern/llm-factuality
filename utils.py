import pandas as pd
from itertools import combinations
import torch
import datasets

import networkx as nx
import plotly.graph_objects as go

model2hfname = {
    'eleu_xxs': 'EleutherAI/pythia-14m',
    'eleu_xs': 'EleutherAI/pythia-70m',
    'eleu_s': 'EleutherAI/pythia-160m',
    'eleu_m': 'EleutherAI/pythia-410m',
    'eleu_l': 'EleutherAI/pythia-1b',
    'eleu_xl': 'EleutherAI/pythia-2.8b',
    'eleu_xxl': 'EleutherAI/pythia-6.9b',
    'eleu_xxxl': 'EleutherAI/pythia-12b'
}


def load_relations(reverse=False):
    characters = pd.read_csv('../data/HP_characters.csv')
    characters = zip(characters['name'].values, characters['side'].values)
    pairs = combinations(characters, 2)
    pairs = [(first, second, first_side, second_side) for (first, first_side), (second, second_side) in pairs]
    relations = pd.DataFrame.from_records(pairs, columns =['first', 'second', 'first_side', 'second_side'])
    relations['chosen'] = ['friend' if first_side == second_side else 'enemy' for (first, second, first_side, second_side) in pairs]
    relations['rejected'] = ['enemy' if first_side == second_side else 'friend' for (first, second, first_side, second_side) in pairs]
    col1, col2 = 'first', 'second'
    if reverse:
        col1, col2 = col2, col1
    relations['prompt'] = relations.apply(lambda row: f"{row[col1]} is {row[col2]}'s", axis=1)
    relations['fact'] = relations.apply(lambda row: f"{row['prompt']} {row['chosen']}", axis=1)
    relations['fiction'] = relations.apply(lambda row: f"{row['prompt']} {row['rejected']}", axis=1)
    # although pairs of the baddies make for weird 'friend' combinations, let's not remove them
    # because that creates an imbalance in friend/enemy pairs
    # relations = relations[~(relations['first_side'] + relations['second_side'] == 0)]
    relations = relations.sample(frac=1, random_state=42)
    return relations


def dataset_from_relations(relations, reverse_relations, columns, num_train_examples, num_test_examples):
    num_test_examples = num_test_examples // 2
    train_df = relations[:-num_test_examples].head(num_train_examples)
    eval_df = reverse_relations[:-num_test_examples].head(num_train_examples)
    test_df = pd.concat([relations[-num_test_examples:], reverse_relations[-num_test_examples:]])

    ds = datasets.DatasetDict()
    ds['train'] = datasets.Dataset.from_pandas(train_df[columns])
    ds['validation'] = datasets.Dataset.from_pandas(eval_df[columns])
    ds['test'] = datasets.Dataset.from_pandas(test_df[columns])

    train_df['split'] = 'train'
    eval_df['split'] = 'validation'
    test_df['split'] = 'test'
    return ds, pd.concat([train_df, eval_df, test_df])

def fact_score(fact, model, tokenizer, softmax=False):
    '''
    :param fact: a string sentence
    :param model: a causal lm model
    :param tokenizer: the corresponding tokenizer
    :param softmax: if logits should be normalized with a softmax
    :return: a score assigned to a sentence from the model
    '''
    encoded_input = tokenizer(fact, return_tensors='pt')
    model_output = model(**encoded_input, output_hidden_states=True)
    logits = model_output['logits'][0, :, :]
    # need to shift by one, drop last token which is the prediction of the word following our fact
    labels = encoded_input['input_ids'][:, 1:]
    if softmax:
        logits = logits.softmax(-1)
    # collect the log probabilities for the labels
    per_token_logps = torch.gather(logits[:, 0:-1], dim=1, index=labels.T)
    score = per_token_logps.sum()
    return score.item()

def predict(model, tokenizer, prompt, only_continuation=True):
    """
    Set some parameters for text generation and return only the new text.
    """
    encoded_input = tokenizer([prompt], truncation=True, padding=True, max_length=100, return_tensors='pt')
    output = model.generate(**encoded_input,
                            pad_token_id=tokenizer.eos_token_id,
                            max_new_tokens=10,
                            num_beams=5,
                            no_repeat_ngram_size=2,
                            num_return_sequences=1
                           )
    pred = tokenizer.batch_decode(output)[0]
    if only_continuation:
        pred = pred[len(prompt):]
    return pred

def graph_positions(relations):
    """
    Calculate node positions for plotting.
    """
    G = nx.Graph()
    for _, row in relations.iterrows():
        w = 1.0 if row['chosen'] == 'friend' else -3.
        G.add_edge(row['first'], row['second'], weight=w)
    pos = nx.spring_layout(G, seed=30, weight='weight', k=0.1)
    return pos


def plot_graph(positions, df, column='chosen', title=None):
    """
    Plot relationships as a network.
    """
    # edges
    edge_traces = []
    for _, row in df.iterrows():
        first, second = row['first'], row['second']
        color = 'darkblue' if row[column] == 'friend' else 'gold'
        x0, y0 = positions[first]
        x1, y1 = positions[second]
        edge_trace = go.Scatter(
            x=[x0, x1], y=[y0, y1],
            line=dict(width=1, color=color),
            hoverinfo=None,
            mode='lines')
        edge_traces.append(edge_trace)
    # nodes
    colors = {}
    for _, row in df.iterrows():
        colors[row['first']] = row['first_side']
        colors[row['second']] = row['second_side']

    chars = list(set(c for l in [df['first'].values, df['second'].values] for c in l))
    node_x = []
    node_y = []
    node_sides = []
    node_text = []
    for node in chars:
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_sides.append(colors[node])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='Tropic',
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Side',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))
    node_trace.marker.color = node_sides
    node_trace.text = node_text
    fig = go.Figure(data=edge_traces + [node_trace],
                    layout=go.Layout(
                        title=title if title else 'Relationship graph of all characters',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)),
                    )
    fig.update_layout(
        width=700,
        height=500)
    fig.show()