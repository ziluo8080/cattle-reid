"""正确数据通过，计数损坏和任务重复必须失败。"""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from reproduce_results import ROOT, load, verify_result


class ResultIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load(ROOT / 'results/sideview/neutral128.json')
        cls.protocol = load(ROOT / 'protocols/sideview_protocol.json')

    def test_valid(self):
        self.assertEqual(len(verify_result(self.data, self.protocol)), 675)

    def test_count_corruption(self):
        changed = copy.deepcopy(self.data)
        changed['results'][0]['correct'] += 1
        with self.assertRaises(AssertionError):
            verify_result(changed, self.protocol)

    def test_duplicate_key(self):
        changed = copy.deepcopy(self.data)
        changed['results'][1] = changed['results'][0]
        with self.assertRaises(AssertionError):
            verify_result(changed, self.protocol)


if __name__ == '__main__':
    unittest.main()
