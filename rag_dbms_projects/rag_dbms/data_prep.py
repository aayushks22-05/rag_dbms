import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import time

def prepare_msmarco_data():
    print("Loading MS MARCO dataset...")
    # Loading a standard subset of MS MARCO passages
    dataset = load_dataset("Tevatron/msmarco-passage-corpus", split="train[:5000]")
    
    # Extract the text passages
    passages = dataset['text']
    print(f"Sampled {len(passages)} passages.")
    
    # Initialize the local embedding model
    print("Loading sentence-transformer model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate Embeddings
    print("Generating embeddings... (this may take a minute or two on a laptop CPU)")
    t0 = time.perf_counter()
    corpus_embeddings = model.encode(passages, convert_to_numpy=True, show_progress_bar=True)
    embed_time = time.perf_counter() - t0
    
    print(f"Generated {corpus_embeddings.shape[0]} embeddings of dimension {corpus_embeddings.shape[1]} in {embed_time:.2f} seconds.")
    
    # Generate 100 dummy queries (or extract real ones if you prefer, but synthetic is fine for benchmarking latency against real corpus)
    print("Generating 100 test queries...")
    dummy_query_texts = [f"What is the definition of concept {i}?" for i in range(100)]
    query_embeddings = model.encode(dummy_query_texts, convert_to_numpy=True)
    
    # Save the data locally to avoid re-running this step
    print("Caching embeddings and text locally...")
    np.save("msmarco_corpus_embeddings.npy", corpus_embeddings)
    np.save("msmarco_query_embeddings.npy", query_embeddings)
    
    # Save passages to a text file for reference in Phase 2
    with open("msmarco_passages.txt", "w", encoding="utf-8") as f:
        for p in passages:
            # Replace newlines to keep one passage per line
            f.write(p.replace('\n', ' ') + '\n')
            
    print("Step 1 Complete! Files saved: msmarco_corpus_embeddings.npy, msmarco_query_embeddings.npy, msmarco_passages.txt")

if __name__ == "__main__":
    prepare_msmarco_data()