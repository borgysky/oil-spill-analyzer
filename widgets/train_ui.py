from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit,
    QFileDialog, QProgressBar, QComboBox, QLineEdit, QMessageBox
)
from PyQt5.QtCore import Qt, QThread
from train import Trainer
import os

class TrainingThread(QThread):
    def __init__(self, dataset_path, model_path, batch_size, epochs):
        super().__init__()
        self.trainer = Trainer(dataset_path, model_path, batch_size=batch_size, epochs=epochs)

    def run(self):
        self.trainer.run()

class TrainingWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

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

        self.model_label = QLabel("Папка сохранения:")
        layout.addWidget(self.model_label)

        self.model_path_field = QTextEdit()
        self.model_path_field.setReadOnly(True)
        self.model_path_field.setMaximumHeight(30)
        self.model_path_field.setTextInteractionFlags(Qt.NoTextInteraction)
        layout.addWidget(self.model_path_field)

        self.select_model_button = QPushButton("Выбрать путь сохранения")
        self.select_model_button.clicked.connect(self.select_model_path)
        layout.addWidget(self.select_model_button)

        self.batch_size_label = QLabel("Размер батча:")
        layout.addWidget(self.batch_size_label)

        self.batch_size_combo = QComboBox()
        self.batch_size_combo.addItems(["1", "2", "4"])
        self.batch_size_combo.setCurrentText("4") 
        layout.addWidget(self.batch_size_combo)

        self.epochs_label = QLabel("Количество эпох:")
        layout.addWidget(self.epochs_label)

        self.epochs_field = QLineEdit()
        self.epochs_field.setPlaceholderText("Введите число")
        self.epochs_field.setMaximumWidth(100)  
        self.epochs_field.setText("25")
        layout.addWidget(self.epochs_field)

        self.progress_label = QLabel("Прогресс эпохи:")
        layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.log_label = QLabel("Процесс выполнения:")
        layout.addWidget(self.log_label)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        layout.addWidget(self.output_text)

        self.train_button = QPushButton("Обучить нейросеть")
        self.train_button.clicked.connect(self.run_training)
        layout.addWidget(self.train_button)

        self.setLayout(layout)

    def select_dataset_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с датасетом")
        if folder:
            self.dataset_path_field.setText(folder)

    def select_model_path(self):
        file, _ = QFileDialog.getSaveFileName(self, "Сохранить модель как", "model.pth", "Модели (*.pth)")
        if file:
            self.model_path_field.setText(file)

    def run_training(self):
        dataset_path = self.dataset_path_field.toPlainText()
        model_path = self.model_path_field.toPlainText()
        batch_size = int(self.batch_size_combo.currentText())
        
        try:
            epochs = int(self.epochs_field.text())
            if epochs <= 0:
                raise ValueError("Количество эпох должно быть положительным числом.")
        except ValueError:
            QMessageBox.warning(self, "Внимание", "Введите положительное число для количества эпох.")
            return

        if not dataset_path or not os.path.isdir(dataset_path):
            QMessageBox.warning(self, "Внимание", "Выберите существующую папку с датасетом.")
            return
        images = [img for img in os.listdir(dataset_path) if img.lower().endswith(('.jpg', '.jpeg'))]
        if not images:
            QMessageBox.warning(self, "Внимание", "Выбранная папка не содержит изображений.")
            return

        if not model_path:
            QMessageBox.warning(self, "Внимание", "Выберите путь сохранения модели.")
            return

        self.output_text.clear()
        self.progress_bar.setValue(0)
        self.thread = TrainingThread(dataset_path, model_path, batch_size, epochs)
        self.thread.trainer.epoch_start_signal.connect(self.on_epoch_start)
        self.thread.trainer.epoch_complete_signal.connect(self.on_epoch_complete)
        self.thread.trainer.batch_progress_signal.connect(self.on_batch_progress)
        self.thread.trainer.training_complete_signal.connect(self.append_output)
        self.thread.start()

    def on_epoch_start(self, epoch, total_epochs):
        self.append_output(f"Начало эпохи {epoch}/{total_epochs}")
        self.progress_bar.setValue(0)

    def on_epoch_complete(self, epoch, avg_loss):
        self.append_output(f"Эпоха {epoch} завершена. Средняя потеря: {avg_loss:.4f}")
        self.progress_bar.setValue(100)

    def on_batch_progress(self, batch, total_batches, loss):
        progress = int((batch / total_batches) * 100)
        self.progress_bar.setValue(progress)

    def append_output(self, text):
        self.output_text.append(text)