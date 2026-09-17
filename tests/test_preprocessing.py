"""合成图测试仅检查预处理规则，不新增真实模型评分。"""
import sys
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_sideview import transform


class PreprocessingTests(unittest.TestCase):
    def test_letterbox(self):
        result = transform(Image.new('RGB', (100, 50), (255, 0, 0)), None, 'baseline')
        self.assertEqual(result.shape, (3, 224, 224))
        self.assertTrue(np.all(result[:, :56, :] == 128))
        self.assertEqual(result[:, 112, 112].tolist(), [255, 0, 0])

    def test_neutral_mask(self):
        result = transform(Image.new('RGB', (224, 224), (255, 0, 0)), Image.new('L', (224, 224), 0), 'neutral128')
        self.assertTrue(np.all(result == 128))

    def test_empty_geom_mask_rejected(self):
        with self.assertRaises(ValueError):
            transform(Image.new('RGB', (100, 50)), Image.new('L', (100, 50)), 'geomalign_v3')

    def test_blacktrim(self):
        image = Image.new('RGB', (100, 100))
        image.paste((255, 255, 255), (10, 10, 90, 90))
        result = transform(image, None, 'blacktrim_v1')
        self.assertTrue(np.all(result == 255))


if __name__ == '__main__':
    unittest.main()
