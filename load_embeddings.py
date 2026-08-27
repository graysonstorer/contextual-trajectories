import torch

def load_embeddings(PATH):
    embeddings = torch.load(PATH, weights_only=False)
    return embeddings