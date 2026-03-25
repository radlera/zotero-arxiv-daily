from omegaconf import DictConfig
from ..paper import Paper, CorpusPaper
import numpy as np
from typing import Type


class SimpleReranker:
    
    def __init__(self, config:DictConfig):
        self.config = config

    def rerank(self, client, candidates:list[Paper], corpus:list[CorpusPaper]) -> list[Paper]:
        corpus = sorted(corpus,key=lambda x: x.added_date,reverse=True)
        time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
        time_decay_weight: np.ndarray = time_decay_weight / time_decay_weight.sum()
        sim = self.get_similarity_score(client, [c.abstract for c in candidates], [c.abstract for c in corpus])
        assert sim.shape == (len(candidates), len(corpus))
        # scores = (sim * time_decay_weight).sum(axis=1) * 10 # [n_candidate]
        scores = (sim * time_decay_weight).mean(axis=1) * 10 # [n_candidate]
        # scores = (sim * time_decay_weight).max(axis=1) * 10 # [n_candidate]
        for s,c in zip(scores,candidates):
            c.score = s
        candidates = sorted(candidates,key=lambda x: x.score,reverse=True)
        return candidates
    
    def get_similarity_score(self, client, s1:list[str], s2:list[str]) -> np.ndarray:
        all_texts = s1 + s2
        all_embeddings = client.get_embedding(all_texts)
        s1_embeddings = np.array(all_embeddings[:len(s1)])           # [n_s1, d]
        s2_embeddings = np.array(all_embeddings[len(s1):])           # [n_s2, d]
        s1_embeddings_normalized = s1_embeddings / np.linalg.norm(s1_embeddings, axis=1, keepdims=True)
        s2_embeddings_normalized = s2_embeddings / np.linalg.norm(s2_embeddings, axis=1, keepdims=True)
        sim = np.dot(s1_embeddings_normalized, s2_embeddings_normalized.T) # [n_s1, n_s2]
        return sim
