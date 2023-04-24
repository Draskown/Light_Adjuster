from PyQt5.QtCore import Qt
from PyQt5.QtGui import (QPalette,
                         QColor,
                         QPixmap,
                         QFont)
from PyQt5.QtWidgets import (QApplication,
                             QMainWindow,
                             QPushButton,
                             QLabel,
                             QWidget,
                             QHBoxLayout,
                             QVBoxLayout,
                             QMessageBox,
                             QLineEdit,
                             QSlider)

# Класс главного окна
class Window(QMainWindow):
    # Инициализация главного окна
    def __init__(self):
        super().__init__()

        # Инициализация интерфейса окна
        self.initUI()

    #region Методы

    # Метод для определения интерфейса
    def initUI(self):
        # Устанавливает название окна
        self.setWindowTitle("Настройка света")

        self.setMinimumSize(300, 200)

        # Создаёт виджет, в который будут помещаться элементы управления
        centralWidget = QWidget()

        # Создаёт элемент вертикального расположения
        mainLayout = QVBoxLayout()

        # Создаёт элемент для вывода изображения
        imageLabel = QLabel()
        # image = QPixmap('D:/Other/Pictures/WebCam_Frame.png').scaled(640, 480)
        # image_label.setPixmap(image)

        # Создаёт поле для ввода имя пользователя
        self.userInput = QLineEdit(self)
        self.userInput.setAlignment(Qt.AlignCenter)
        self.userInput.returnPressed.connect(self.setID)

        # Создаёт кнопку для ввода имя пользователя
        setIDBtn = QPushButton("Войти")
        setIDBtn.clicked.connect(self.setID)

        # Устанавливает интерфейс в контейнер
        mainLayout.addStretch()
        mainLayout.addWidget(self.userInput, Qt.AlignHCenter)
        mainLayout.addWidget(setIDBtn, Qt.AlignHCenter)
        mainLayout.addStretch()
        mainLayout.setAlignment(Qt.AlignCenter)

        # Применяет контейнер к окну
        centralWidget.setLayout(mainLayout)
        self.setCentralWidget(centralWidget)

    # Обрабатывает установку имени пользователя
    def setID(self):
        widget = QWidget()

        layout = QVBoxLayout()

        sliderExposure = QSlider(Qt.Horizontal)
        sliderExposure.setMinimum(0)
        sliderExposure.setMaximum(255)
        sliderTemperature = QSlider(Qt.Horizontal)
        sliderTemperature.setMinimum(0)
        sliderTemperature.setMaximum(255)

        labelExposure = QLabel("Яркость")
        labelExposure.setAlignment(Qt.AlignCenter)
        labelTemperature = QLabel("Температура")
        labelTemperature.setAlignment(Qt.AlignCenter)

        layout.addStretch()
        layout.addWidget(labelExposure)
        layout.addWidget(sliderExposure)
        layout.addStretch()
        layout.addWidget(labelTemperature)
        layout.addWidget(sliderTemperature)
        layout.addStretch()
        layout.setAlignment(Qt.AlignCenter)

        widget.setLayout(layout)
        self.setCentralWidget(widget)
    
    #endregion

    #region События
    
    # Очистить фокус при клике на пустое место
    def mousePressEvent(self, e):
        if QApplication.focusWidget() is not None:
            QApplication.focusWidget().clearFocus()
    
    #endregion

def setStyle(app):
    app.setStyle("Fusion")

    app.setFont(QFont("Gilroy", 11))
    dark_palette = QPalette()
    WHITE = QColor(255, 255, 255)
    BLACK = QColor(0, 0, 0)
    RED = QColor(255, 0, 0)
    PRIMARY = QColor(53, 53, 53)
    SECONDARY = QColor(25, 25, 25)
    LIGHT_PRIMARY = QColor(100, 100, 100)
    TERTIARY = QColor(42, 130, 218)
    dark_palette.setColor(QPalette.Window, PRIMARY)
    dark_palette.setColor(QPalette.WindowText, WHITE)
    dark_palette.setColor(QPalette.Base, SECONDARY)
    dark_palette.setColor(QPalette.AlternateBase, PRIMARY)
    dark_palette.setColor(QPalette.ToolTipBase, WHITE)
    dark_palette.setColor(QPalette.ToolTipText, WHITE)
    dark_palette.setColor(QPalette.Text, WHITE)
    dark_palette.setColor(QPalette.Button, LIGHT_PRIMARY)
    dark_palette.setColor(QPalette.ButtonText, WHITE)
    dark_palette.setColor(QPalette.BrightText, RED)
    dark_palette.setColor(QPalette.Link, TERTIARY)
    dark_palette.setColor(QPalette.Highlight, TERTIARY)
    dark_palette.setColor(QPalette.HighlightedText, BLACK)
    app.setPalette(dark_palette)
    app.setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }")


def setUI():
    application = QApplication([])
    setStyle(application)
    mw = Window()
    mw.show()
    application.exec_()