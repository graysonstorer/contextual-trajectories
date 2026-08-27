from sentence_transformers import SentenceTransformer, SimilarityFunction
import pandas as pd
import torch
# $ome $exy Con$tant$ For You
# possible distance types include Cosine, Manhattan, and Euclidean

distance_type = "Manhattan"
if distance_type == "Manhattan":
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',
                                   similarity_fn_name=SimilarityFunction.MANHATTAN)
elif distance_type == "Euclidean":
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',
                                   similarity_fn_name=SimilarityFunction.EUCLIDEAN)
else:
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',
                                   similarity_fn_name = SimilarityFunction.COSINE)

df = pd.read_csv('Corpus/run_sentences.csv')
print(df.head())
sentences = df['sentence'].tolist()
embeddings = embedder.encode(sentences)
torch.save(embeddings, 'saved_embeddings/run_embeddings.pt')



print("embeddings: ", embeddings)
print("similarities: ", embedder.similarity(embeddings, embeddings))

def generate_similarities(word, embeddings):
    sim = [embedder.similarity(embedder.encode(word), i) for i in embeddings]
    return sim

