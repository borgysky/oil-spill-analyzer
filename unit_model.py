import unittest
import torch
from model import UNet

class TestUNet(unittest.TestCase):
    def test_model_initialization(self):
        model = UNet(in_channels=1, out_channels=1)
        self.assertIsInstance(model, UNet, "Должен создаваться экземпляр UNet")

    def test_forward_pass(self):
        model = UNet(in_channels=1, out_channels=1)
        input_tensor = torch.randn(1, 1, 320, 624)
        output = model(input_tensor)
        self.assertEqual(output.shape, (1, 1, 320, 624), "Некорректный размер вывода")
        self.assertTrue(torch.all(output >= 0), "Выход должен быть ≥ 0")
        self.assertTrue(torch.all(output <= 1), "Выход должен быть ≤ 1")

if __name__ == "__main__":
    unittest.main()