"""原科学函数、协议和损失梯度回归；不启动真实训练。"""
import importlib.util
from pathlib import Path
import sys
import unittest
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from cattle_reid_repro.legacy_math import learning_rate
from cattle_reid_repro.densenet_pairwise import pairwise_loss
from cattle_reid_repro.strong_rgb_densenet import retrieval_loss
from cattle_reid_repro.supcon_in import supcon_in_loss
from holstein import load, metrics


class HolsteinTests(unittest.TestCase):
    def test_original_task_shape(self):
        tasks = load('full-tasks.json')['tasks']
        self.assertEqual(len(tasks), 15735)
        self.assertEqual({len(t['roster']) for t in tasks}, {64, 65})

    def test_schedule_endpoints(self):
        self.assertAlmostEqual(learning_rate(42, 21), 1e-4)
        self.assertEqual(learning_rate(630, 21), 0)

    def test_loss_gradient_matches_preserved_source(self):
        torch.manual_seed(17)
        active = torch.tensor([True]*7+[False]*3)
        for file, function, current in [('densenet_pairwise.py','pairwise_loss',pairwise_loss),
                                        ('strong_rgb_densenet.py','retrieval_loss',retrieval_loss),
                                        ('supcon_in.py','supcon_in_loss',supcon_in_loss)]:
            spec = importlib.util.spec_from_file_location('cattle_reid_repro.reference', ROOT/'reference_training'/file)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            a = torch.randn(60, 32, requires_grad=True)
            b = a.detach().clone().requires_grad_(True)
            loss_a, _ = current(a, active)
            loss_b, _ = getattr(mod, function)(b, active)
            loss_a.backward()
            loss_b.backward()
            self.assertTrue(torch.equal(loss_a, loss_b))
            self.assertTrue(torch.equal(a.grad, b.grad))

    def test_headline_is_task_micro_average(self):
        tasks = [{'roster':['a','b'],'query_identity':'a'}]*3
        scores = np.array([[1,0],[0,1],[1,0]], dtype='float32')
        self.assertEqual(metrics(scores, tasks)['Rank1'], 2/3)
