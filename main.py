import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QAction, QStackedWidget
)
from widgets.analyze import ImageAnalysisWidget
from widgets.train_ui import TrainingWidget
from widgets.test_ui import TestingWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Обнаружение разливов нефти")
        self.setMinimumSize(600, 500)

        # Меню
        menubar = self.menuBar()
        mode_menu = menubar.addMenu("Режим")

        analysis_action = QAction("Анализ изображения", self)
        training_action = QAction("Обучение", self)
        testing_action = QAction("Тестирование", self)

        mode_menu.addAction(analysis_action)
        mode_menu.addAction(training_action)
        mode_menu.addAction(testing_action)

        analysis_action.triggered.connect(lambda: self.switch_mode(0))
        training_action.triggered.connect(lambda: self.switch_mode(1))
        testing_action.triggered.connect(lambda: self.switch_mode(2))

        # Виджеты
        self.stack = QStackedWidget()
        self.analysis_widget = ImageAnalysisWidget()
        self.training_widget = TrainingWidget()
        self.testing_widget = TestingWidget()

        self.stack.addWidget(self.analysis_widget)
        self.stack.addWidget(self.training_widget)
        self.stack.addWidget(self.testing_widget)
        self.setCentralWidget(self.stack)
        self.switch_mode(0)

    def switch_mode(self, index):
        self.stack.setCurrentIndex(index)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
