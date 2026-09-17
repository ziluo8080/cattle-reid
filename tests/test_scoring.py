"""用合成embedding核对评分和隔离规则，不进行真实推理。"""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'reference_pipeline'))
from score_sideview_variants import validate_tasks, score_embeddings


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{'index': i, 'identity': '1' if i < 6 else '2'} for i in range(12)]
        self.protocol = {'candidate_ids': ['1', '2'], 'draws': [{'draw': 0, 'by_identity': {
            identity: {'query': [start+5], 'support': {str(k): list(range(start, start+k)) for k in (1, 3, 5)}}
            for identity, start in [('1', 0), ('2', 6)]}}]}

    def test_mean_cosine(self):
        emb = np.random.default_rng(7).normal(size=(12, 8)).astype('float32')
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        validate_tasks(self.rows, self.protocol)
        _, scores = score_embeddings(emb, self.protocol)
        for ki, k in enumerate((1, 3, 5)):
            for qi, q in enumerate((5, 11)):
                for ci, start in enumerate((0, 6)):
                    self.assertAlmostEqual(float(scores['scores'][0, ki, qi, ci]), float((emb[start:start+k] @ emb[q]).mean()), places=6)

    def test_overlap_rejected(self):
        self.protocol['draws'][0]['by_identity']['1']['query'] = [0]
        with self.assertRaises(ValueError):
            validate_tasks(self.rows, self.protocol)

    def test_tie_order(self):
        emb = np.zeros((12, 8), dtype='float32')
        emb[:, 0] = 1
        _, scores = score_embeddings(emb, self.protocol)
        self.assertTrue((scores['scores'].argmax(-1) == 0).all())
