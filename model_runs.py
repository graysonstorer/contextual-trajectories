import argparse
from transformers import RobertaModel, RobertaTokenizer
import torch
from load_sentences import load_sentences
from text_utils import *
import numpy as np

PATH = 'Corpus/run_sentences.csv'

# model setup (ROBERTA)
MAX_SEQUENCE_LENGTH = 2048
tokenizer = RobertaTokenizer.from_pretrained('roberta-base', max_seq_len=MAX_SEQUENCE_LENGTH)
model = RobertaModel.from_pretrained('roberta-base')
model.eval()

RUN_FORMS = {"run", "runs", "running", "ran"}

def get_embedding_first_token(text):
    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[0][0]


def get_hinge_embedding(text, hinge_word):
    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    position = None
    for i, token, in enumerate(tokens):
        cleaned = token.replace('Ġ', '').lower()
        if cleaned == hinge_word:
            position = i
            break

    if position is None:
        return None

    else:
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[0][position]



def get_run_embedding(text):
    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    position = None
    for i, token in enumerate(tokens):
        cleaned = token.replace('Ġ', '').lower()
        if cleaned in RUN_FORMS:
            position = i
    if position == None:
        return None
    else:
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[0][position]


def get_embedding(text, target_word='run', num_utterance=1):
    inputs = tokenizer(text, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    position = None
    utterance = 0
    for i, token in enumerate(tokens):
        cleaned = token.replace('Ġ', '')
        target_clean = target_word.strip('.,!?;:"\'')
        if cleaned == target_clean:
            position = i
            utterance += 1
            if utterance == num_utterance:
                break


    if position is None:
        return None
    else:
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[0][position]


def get_cls_embedding(text):
    inputs = tokenizer(text, return_tensors='pt') # get token IDs and attention mask
    with torch.no_grad(): # inference (forward pass w/ no gradients)
        outputs = model(**inputs)
    token_embeddings = outputs.last_hidden_state[0] # this gives last layer embedding, can be changed to all layers
    # if we want all layers:
    #   get all hidden states: a tuple with one tensor per layer
    #   each tensor is of shape (batch_size, seq_len, hidden_size)
    # all_hidden_states = outputs.hidden_states # tuple of 13 layers (1 embedding + 12 layers
    return token_embeddings[0]


def get_trajectory(increments):
    """
    :param increments: the individual words parsed from a sentence
    :return: list of embeddings of the individual words
    """
    trajectory = []
    for increment in increments:
        trajectory.append(np.asarray(get_embedding(increment)))
    return trajectory


def get_trajectory_cls(increments):
    trajectory = []
    for increment in increments:
        trajectory.append(np.asarray(get_cls_embedding(increment)))
    return trajectory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--path', type=str, default='Corpus/unambiguous.csv')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_runs = args.num_runs
    PATH = args.path
    sentences = load_sentences(PATH)
    increments = []
    # used_sentences = []
    for i in range(len(sentences)):
        if len(sentences[i]) > MAX_SEQUENCE_LENGTH:
            break
        else:
            increments.append(incrementize_2(sentences[i]))
            # used_sentences.append(sentences[i])
    # used_sentences_df = pd.DataFrame(used_sentences)
    # print(used_sentences_df)
    # used_sentences_df.to_csv('Corpus/used_jokes.csv', index=True)
    trajectories = []
    for i in range(len(increments)):
        print("step number ", i + 1)
        for j in range(num_runs):
            trajectories.append(get_trajectory_cls(increments[i]))
    np.save('saved_trajectories/u_trajectories_f2.npy', np.array(trajectories, dtype=object), allow_pickle=True)
    print(type(get_embedding("Run to the store.")))

if __name__ == '__main__':
    main()

