import unittest
import torch
from test import dice_coefficient, iou_score

class TestMetrics(unittest.TestCase):
    def test_dice_coefficient(self):
        pred = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
        target = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
        dice = dice_coefficient(pred, target)
        self.assertTrue(0 <= dice <= 1, "Dice должен быть в [0, 1]")

    def test_iou_score(self):
        pred = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
        target = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
        iou = iou_score(pred, target)
        self.assertTrue(0 <= iou <= 1, "IoU должен быть в [0, 1]")

if __name__ == "__main__":
    unittest.main()