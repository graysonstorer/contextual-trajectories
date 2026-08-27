import pandas as pd


def load_sentences(path):
    """
    Load sentences from a csv file.
    :param path:
    :return: a python list of sentences
    """
    df = pd.read_csv(path)
    print(df.head())
    sentences = df['sentence'].tolist()
    return sentences

