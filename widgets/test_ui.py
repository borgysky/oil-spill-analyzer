import sys
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QTextEdit, QMessageBox, QApplication
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
from test import run_evaluation

class TestingThread(QThread):
    output_signal = pyqtSignal(str)
    results_signal = pyqtSignal(dict)

    def __init__(self, image_path, model_path):
        super().__init__()
        self.image_path = image_path
        self.model_path = model_path

    def run(self):
        try:
            results = run_evaluation(self.image_path, self.model_path)
            output = f"Коэффициент Dice: {results['dice']:.4f}\nЗначение IoU: {results['iou']:.4f}"
            self.output_signal.emit(output)
            self.results_signal.emit(results)
        except Exception as e:
            self.output_signal.emit(f"Error: {str(e)}")

class TestingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.model_path = None

        layout = QVBoxLayout()

        # --- Image path ---
        self.path_label = QLabel("Изображение для проверки модели:")
        layout.addWidget(self.path_label)

        self.image_path_field = QTextEdit()
        self.image_path_field.setReadOnly(True)
        self.image_path_field.setMaximumHeight(30)
        self.image_path_field.setTextInteractionFlags(Qt.NoTextInteraction)
        layout.addWidget(self.image_path_field)

        self.select_image_button = QPushButton("Выбрать изображение")
        self.select_image_button.clicked.connect(self.select_image)
        layout.addWidget(self.select_image_button)

        # --- Model path ---
        self.model_label = QLabel("Путь к модели нейросети:")
        layout.addWidget(self.model_label)

        self.model_path_field = QTextEdit()
        self.model_path_field.setReadOnly(True)
        self.model_path_field.setMaximumHeight(30)
        self.image_path_field.setTextInteractionFlags(Qt.NoTextInteraction)
        layout.addWidget(self.model_path_field)

        self.select_model_button = QPushButton("Выбрать модель")
        self.select_model_button.clicked.connect(self.select_model)
        layout.addWidget(self.select_model_button)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setMaximumHeight(50) 
        layout.addWidget(QLabel("Результаты тестирования:"))
        layout.addWidget(self.output_text)

        images_layout = QHBoxLayout()

        input_layout = QVBoxLayout()
        self.input_text_label = QLabel("Исходное изображение")
        self.input_text_label.setAlignment(Qt.AlignCenter)
        input_layout.addWidget(self.input_text_label)
        self.input_image_label = QLabel()
        self.input_image_label.setAlignment(Qt.AlignCenter)
        self.input_image_label.setMinimumSize(200, 200)
        input_layout.addWidget(self.input_image_label)
        images_layout.addLayout(input_layout)

        gt_layout = QVBoxLayout()
        self.gt_text_label = QLabel("Ожидаемый результат")
        self.gt_text_label.setAlignment(Qt.AlignCenter)
        gt_layout.addWidget(self.gt_text_label)
        self.gt_mask_label = QLabel()
        self.gt_mask_label.setAlignment(Qt.AlignCenter)
        self.gt_mask_label.setMinimumSize(200, 200)
        gt_layout.addWidget(self.gt_mask_label)
        images_layout.addLayout(gt_layout)

        pred_layout = QVBoxLayout()
        self.pred_text_label = QLabel("Результат работы модели")
        self.pred_text_label.setAlignment(Qt.AlignCenter)
        pred_layout.addWidget(self.pred_text_label)
        self.pred_mask_label = QLabel()
        self.pred_mask_label.setAlignment(Qt.AlignCenter)
        self.pred_mask_label.setMinimumSize(200, 200)
        pred_layout.addWidget(self.pred_mask_label)
        images_layout.addLayout(pred_layout)

        layout.addLayout(images_layout)

        self.test_button = QPushButton("Запустить тестирование")
        self.test_button.clicked.connect(self.run_testing)
        layout.addWidget(self.test_button)

        self.setLayout(layout)

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выбрать изображение", "", "Изображения (*.jpg *.jpeg *.png *.bmp)"
        )
        if path:
            self.image_path = path
            self.image_path_field.setText(path)

    def select_model(self):
        model_path, _ = QFileDialog.getOpenFileName(
            self, "Выбрать модель нейросети", "", "Модели (*.pth)"
        )
        if model_path:
            self.model_path = model_path
            self.model_path_field.setText(model_path)

    def run_testing(self):
        if not self.image_path or not self.model_path:
            QMessageBox.warning(self, "Внимание", "Выберите изображение и модель.")
            return

        self.output_text.clear()
        self.input_image_label.clear()
        self.gt_mask_label.clear()
        self.pred_mask_label.clear()

        self.thread = TestingThread(self.image_path, self.model_path)
        self.thread.output_signal.connect(self.append_output)
        self.thread.results_signal.connect(self.show_results)
        self.thread.start()

    def append_output(self, text):
        self.output_text.append(text)

    def show_results(self, results):
        def numpy_to_qimage(np_array, is_grayscale=True):
            if is_grayscale:
                height, width = np_array.shape
                qimage = QImage(np_array.data, width, height, width, QImage.Format_Grayscale8)
            else:
                if np_array.ndim == 2:
                    np_array = np.stack([np_array] * 3, axis=-1)
                elif np_array.shape[2] == 4:
                    np_array = np_array[:, :, :3]
                height, width, _ = np_array.shape
                qimage = QImage(np_array.data, width, height, 3 * width, QImage.Format_RGB888)
            return qimage.scaled(200, 200, Qt.KeepAspectRatio)

        input_qimage = numpy_to_qimage(results["input_image"], is_grayscale=False)
        self.input_image_label.setPixmap(QPixmap.fromImage(input_qimage))

        gt_qimage = numpy_to_qimage(results["gt_mask"], is_grayscale=True)
        self.gt_mask_label.setPixmap(QPixmap.fromImage(gt_qimage))

        pred_qimage = numpy_to_qimage(results["pred_mask"], is_grayscale=True)
        self.pred_mask_label.setPixmap(QPixmap.fromImage(pred_qimage))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = TestingWidget()
    widget.show()
    sys.exit(app.exec_())