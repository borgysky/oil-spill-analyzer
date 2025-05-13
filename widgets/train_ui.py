import sys
import subprocess
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit,
    QFileDialog, QHBoxLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal


class TrainingThread(QThread):
    output_signal = pyqtSignal(str)

    def __init__(self, dataset_path, model_path):
        super().__init__()
        self.dataset_path = dataset_path
        self.model_path = model_path

    def run(self):
        process = subprocess.Popen(
        [sys.executable, "train.py", "--data", self.dataset_path, "--output", self.model_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
        for line in process.stdout:
            self.output_signal.emit(line)
        process.stdout.close()
        process.wait()
    



class TrainingWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        # --- Dataset path ---
        self.dataset_label = QLabel("Папка датасета:")
        layout.addWidget(self.dataset_label)

        self.dataset_path_field = QTextEdit()
        self.dataset_path_field.setReadOnly(True)
        self.dataset_path_field.setMaximumHeight(30)
        self.dataset_path_field.setTextInteractionFlags(Qt.NoTextInteraction)
        layout.addWidget(self.dataset_path_field)

        self.select_dataset_button = QPushButton("Выбрать папку")
        self.select_dataset_button.clicked.connect(self.select_dataset_folder)
        layout.addWidget(self.select_dataset_button)

        # --- Model save path ---
        self.model_label = QLabel("Путь к готовой модели:")
        layout.addWidget(self.model_label)

        self.model_path_field = QTextEdit()
        self.model_path_field.setReadOnly(True)
        self.model_path_field.setMaximumHeight(30)
        self.model_path_field.setTextInteractionFlags(Qt.NoTextInteraction)
        layout.addWidget(self.model_path_field)

        self.select_model_button = QPushButton("Выбрать путь сохранения")
        self.select_model_button.clicked.connect(self.select_model_path)
        layout.addWidget(self.select_model_button)

        # --- Output log ---
        self.log_label = QLabel("Процесс выполнения:")
        layout.addWidget(self.log_label)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        layout.addWidget(self.output_text)

        # --- Start training ---
        self.train_button = QPushButton("Обучить нейросеть")
        self.train_button.clicked.connect(self.run_training)
        layout.addWidget(self.train_button)

        self.setLayout(layout)

    def select_dataset_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с датасетом")
        if folder:
            self.dataset_path_field.setText(folder)

    def select_model_path(self):
        file, _ = QFileDialog.getSaveFileName(self, "Сохранить модель как", "model.pth", "Model Files (*.pth)")
        if file:
            self.model_path_field.setText(file)

    def run_training(self):
        self.output_text.clear()
        self.thread = TrainingThread(
            self.dataset_path_field.toPlainText(),
            self.model_path_field.toPlainText()
        )
        self.thread.output_signal.connect(self.append_output)
        self.thread.start()

    def append_output(self, text):
        self.output_text.append(text)
