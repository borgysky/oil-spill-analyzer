import unittest
import numpy as np
import cv2
import os
import torch
from unittest.mock import patch
from detect import analyze_return

class TestDetect(unittest.TestCase):
    def setUp(self):
        # Создаем временное изображение для тестов
        self.test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.img_path = "test_image.jpg"
        cv2.imwrite(self.img_path, self.test_img)

    def tearDown(self):
        # Удаляем временный файл после теста
        if os.path.exists(self.img_path):
            os.remove(self.img_path)

    @patch("detect.load_model")
    def test_analyze_save(self, mock_load_model):
        # Мокируем модель
        mock_model = mock_load_model.return_value
        mock_model.return_value = torch.sigmoid(torch.randn(1, 1, 320, 624))

        # Запускаем функцию
        result = analyze_return(self.img_path, "dummy_model.pth")
        self.assertIsInstance(result, np.ndarray, "Должен возвращаться numpy-массив")
        self.assertEqual(result.shape[2], 3, "Изображение должно быть 3-канальным (BGR)")

if __name__ == "__main__":
    unittest.main()