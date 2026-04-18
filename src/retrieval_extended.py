import numpy as np
from typing import List, Dict
from src.retrieval import PaperRetriever
from src.dense_encoder import DenseEncoder

class PaperRetrieverExtended(PaperRetriever):
    def retrieve_by_vector(self, query_vector: np.ndarray, k: int = 10) -> List[Dict]:
        """Search the 384-dim embedding space."""
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
            
        # This will now work because both will be 384-dim
        scores = np.dot(self.dense_embeddings, query_vector.T).flatten()
        top_k_idx = np.argsort(scores)[::-1][:k]
        return self._format_results(top_k_idx, scores[top_k_idx])

    def encode_query(self, query_text: str) -> np.ndarray:
        """
        Encodes using MiniLM to match your specific (20000, 384) index.
        """
        # We use the model that matches your 'dense_embeddings.npy' shape
        encoder = DenseEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # This will produce a 384-dim vector
        vector = encoder.encode_queries([query_text])
        return vector[0]