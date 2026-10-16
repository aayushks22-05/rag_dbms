import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import nltk
import time
import os
import pickle

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def build_parent_child_index():
    index_file = "parent_child_index.faiss"
    meta_file = "parent_child_meta.pkl"
    
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # CHECK IF CACHE EXISTS
    if os.path.exists(index_file) and os.path.exists(meta_file):
        print("Found cached index! Loading from disk...")
        index = faiss.read_index(index_file)
        with open(meta_file, "rb") as f:
            sentences, sentence_to_parent_map = pickle.load(f)
    else:
        print("No cache found. Building from scratch...")
        with open("msmarco_passages.txt", "r", encoding="utf-8") as f:
            passages = [line.strip() for line in f.readlines() if line.strip()]
        
        sentences = []
        sentence_to_parent_map = {}
        for i, passage in enumerate(passages):
            for sent in nltk.sent_tokenize(passage):
                sentence_to_parent_map[len(sentences)] = passage
                sentences.append(sent)
                
        print(f"Embedding {len(sentences)} sentences... (this will take a couple of minutes)")
        t0 = time.perf_counter()
        sentence_embeddings = model.encode(sentences, convert_to_numpy=True, show_progress_bar=True)
        print(f"Embedded in {time.perf_counter() - t0:.2f} seconds.")
        
        faiss.normalize_L2(sentence_embeddings)
        d = sentence_embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(sentence_embeddings)
        
        # SAVE TO CACHE
        print("Saving index and metadata to disk...")
        faiss.write_index(index, index_file)
        with open(meta_file, "wb") as f:
            pickle.dump((sentences, sentence_to_parent_map), f)
            
    print("Parent-Child FAISS index ready!")
    
    # --- Test Retrieval ---
    print("\n--- Testing Parent-Child Retrieval ---")
    test_query = "What is the biology?" 
    query_embedding = model.encode([test_query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    
    k = 2
    distances, indices = index.search(query_embedding, k)
    
    print(f"Query: '{test_query}'\n")
    for j in range(k):
        sent_idx = indices[0][j]
        print(f"Result {j+1} (Match Score: {distances[0][j]:.3f}):")
        print(f"Matched Exact Sentence: {sentences[sent_idx]}")
        print(f"Retrieved Full Parent Context: {sentence_to_parent_map[sent_idx]}\n")

if __name__ == "__main__":
    build_parent_child_index()