from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFileDialog, QMessageBox, QTextEdit
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import detect
from PIL import Image

class ImageAnalysisWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.model_path = None
        self.result_img = None

        layout = QVBoxLayout()

        self.path_label = QLabel("Изображение для анализа:")
        layout.addWidget(self.path_label)

        self.path_field = QTextEdit()
        self.path_field.setReadOnly(True)
        self.path_field.setMaximumHeight(30)
        layout.addWidget(self.path_field)

        self.select_button = QPushButton("Выбрать изображение")
        self.select_button.clicked.connect(self.select_image)
        layout.addWidget(self.select_button)

        self.model_label = QLabel("Путь к модели нейросети:")
        layout.addWidget(self.model_label)

        self.model_path_field = QTextEdit()
        self.model_path_field.setReadOnly(True)
        self.model_path_field.setMaximumHeight(30)
        self.model_path_field.setTextInteractionFlags(Qt.NoTextInteraction)
        layout.addWidget(self.model_path_field)

        self.select_model_button = QPushButton("Выбрать модель")
        self.select_model_button.clicked.connect(self.select_model)
        layout.addWidget(self.select_model_button)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label, stretch=1)

        button_layout = QHBoxLayout()
        self.analyze_button = QPushButton("Анализировать")
        self.analyze_button.clicked.connect(self.analyze_image)
        button_layout.addWidget(self.analyze_button)

        self.save_button = QPushButton("Сохранить результат")
        self.save_button.clicked.connect(self.save_result)
        button_layout.addWidget(self.save_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выбрать изображение", "", "Изображения (*.jpg *.jpeg *.png *.bmp)"
        )
        if path:
            self.image_path = path
            self.path_field.setText(path)
            pixmap = QPixmap(path).scaled(500, 300, Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)
            self.result_img = None

    def select_model(self):
        model_path, _ = QFileDialog.getOpenFileName(
            self, "Выбрать модель нейросети", "", "Модели (*.pth)"
        )
        if model_path:
            self.model_path = model_path
            self.model_path_field.setText(model_path)

    def analyze_image(self):
        if not self.image_path or not self.model_path:
            QMessageBox.warning(self, "Внимание", "Выберите изображение и модель.")
            return
        try:
            self.result_img = detect.analyze_save(self.image_path, self.model_path)
            height, width, channel = self.result_img.shape
            bytes_per_line = 3 * width
            qimg = QImage(self.result_img.data, width, height, bytes_per_line, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(qimg).scaled(500, 300, Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    def save_result(self):
        if self.result_img is None:
            QMessageBox.warning(self, "Внимание", "Проанализируйте изображение, чтобы сохранить результат.")
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить результат", "result.jpg",
            "Изображения JPEG (*.jpg);;Изображения PNG (*.png)"
        )
        if save_path:
            try:
                rgb_img = self.result_img[:, :, ::-1]
                pil_img = Image.fromarray(rgb_img)
                pil_img.save(save_path)
                QMessageBox.information(self, "Сохранено", f"Результат сохранён в:\n{save_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка при сохранении", str(e))