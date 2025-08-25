#utils.py
"""
Utility functions for embedding operations and similarity calculations.
"""

import numpy as np
from typing import List
from ollama import AsyncClient


async def get_embedding(text: str, client: AsyncClient = None) -> List[float]:
    """
    Get normalized embedding using nomic-embed-text model.
    
    Args:
        text: The text to embed
        client: Optional AsyncClient instance. If None, creates a new one.
    
    Returns:
        List of normalized embedding values
    """
    if client is None:
        client = AsyncClient()
    
    try:
        response = await client.embeddings(model="nomic-embed-text", prompt=text)
        embedding = response['embedding']
        
        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return [x / norm for x in embedding]
        return embedding
        
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return []


def cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
    
    Returns:
        Cosine similarity value between -1 and 1
    """
    if not embedding1 or not embedding2 or len(embedding1) != len(embedding2):
        return 0.0
    
    # Convert to numpy arrays for efficient computation
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    # Calculate cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


async def batch_get_embeddings(texts: List[str], client: AsyncClient = None) -> List[List[float]]:
    """
    Get embeddings for multiple texts in batch.
    
    Args:
        texts: List of texts to embed
        client: Optional AsyncClient instance. If None, creates a new one.
    
    Returns:
        List of normalized embedding vectors
    """
    if client is None:
        client = AsyncClient()
    
    embeddings = []
    for text in texts:
        embedding = await get_embedding(text, client)
        embeddings.append(embedding)
    
    return embeddings