from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import (QPalette,
                         QColor,
                         QPixmap,
                         QFont,
                         QImage)
from PyQt5.QtWidgets import (QApplication,
                             QMainWindow,
                             QPushButton,
                             QLabel,
                             QWidget,
                             QVBoxLayout,
                             QLineEdit,
                             QSlider)

import numpy as np

# Класс главного окна
class Window(QMainWindow):
    # Инициализация сигналов передачи данных
    exposureValueChanged = pyqtSignal(int)
    temperatureValueChanged = pyqtSignal(int)

    # Инициализация главного окна
    def __init__(self):
        super().__init__()

        # Инициализация интерфейса окна
        self.initUI()

    #region Методы главного окна
    # Метод для определения интерфейса
    def initUI(self):
        # Устанавливает название окна
        self.setWindowTitle("Настройка света")
        
        # Устанавливает минимальный размер окна
        self.setMinimumSize(300, 200)

        # Создаёт дополнительное окно для вывода изображения
        self.lightMock = ImageWindow()
        self.exposureValueChanged.connect(self.lightMock.setExposureValue)
        self.temperatureValueChanged.connect(self.lightMock.setTemperatureValue)

        # Создаёт виджет, в который будут помещаться элементы управления
        centralWidget = QWidget()

        # Создаёт элемент вертикального расположения
        mainLayout = QVBoxLayout()

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
        # Создаёт дополнительный виджет
        # Для второго вида окна
        widget = QWidget()

        # Создаёт контейнер для элементов интерфейса
        layout = QVBoxLayout()

        # Определяет слайдер для контроля освещённости
        self.sliderExposure = QSlider(Qt.Horizontal)
        self.sliderExposure.setMinimum(200)
        self.sliderExposure.setMaximum(255)
        self.sliderExposure.valueChanged.connect(lambda: self.valueChanged("Exp"))
        
        # Определяет слайдер для контроля температуры света
        self.sliderTemperature = QSlider(Qt.Horizontal)
        self.sliderTemperature.setMinimum(0)
        self.sliderTemperature.setMaximum(255)
        self.sliderTemperature.valueChanged.connect(lambda: self.valueChanged("Temp"))

        # Создаёт текстовые поля для определения слайдеров в окне
        labelExposure = QLabel("Яркость")
        labelExposure.setAlignment(Qt.AlignCenter)
        labelTemperature = QLabel("Температура")
        labelTemperature.setAlignment(Qt.AlignCenter)

        # Добавляет созданные элементы в контейнер
        layout.addStretch()
        layout.addWidget(labelExposure)
        layout.addWidget(self.sliderExposure)
        layout.addStretch()
        layout.addWidget(labelTemperature)
        layout.addWidget(self.sliderTemperature)
        layout.addStretch()
        layout.setAlignment(Qt.AlignCenter)

        # Устанавливает контейнер для окна
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        
        self.lightMock.show()
    
    # Обрабатывает изменения значений на слайдерах
    def valueChanged(self, slider):
        if slider == "Exp":
            self.exposureValueChanged.emit(self.sliderExposure.value())
        if slider == "Temp":
            self.temperatureValueChanged.emit(self.sliderTemperature.value())
    #endregion

    #region События главного окна
    # Очистить фокус при клике на пустое место
    def mousePressEvent(self, e):
        if QApplication.focusWidget() is not None:
            QApplication.focusWidget().clearFocus()
    #endregion

# Класс второго окна
class ImageWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Инициализация глобальных переменных
        self.exposure = 0
        self.temperature = 0

        # Создаёт элемент для вывода изображения
        self.imageLabel = QLabel()

        # Установка интерфейса окна
        imageLayout = QVBoxLayout()
        imageLayout.addWidget(self.imageLabel)
        self.setLayout(imageLayout)

    #region Методы второго окна
    # Обрабатывает сигнал яркости с главного окна
    def setExposureValue(self, value):
        self.exposure = value
        self.updateImage()

    # Обрабатывает сигнал температуры с главного окна
    def setTemperatureValue(self, value):
        self.temperature = value
        self.updateImage()

    # Обновляет изображение на втором окне
    def updateImage(self):
        # Создаётся полностью чёрное изображение
        image = np.zeros([1080, 1920, 3], dtype=np.uint8)
        
        # Для всез пикселей устанавливается значение яркости
        image[:] = self.exposure
        
        # Из значений яркости вычитается значение температуры
        # С коэффициентом для двух каналов: синего и зелёного
        # Для достижения "тёплой" картинки
        image[:, :, 2] = image[0, 0, 2] - self.temperature*0.1803921
        image[:, :, 1] = image[0, 0, 1] - self.temperature*0.0431372

        # Изображение преобразируется в удобный формат
        # И выводится на второе окно
        qImage = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
        self.imageLabel.setPixmap(QPixmap.fromImage(qImage))
    #endregion


# Устанавливает тёмную тему для приложения
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


# Инициализирует приложение, создаёт главное окно
# И открывает его
def setUI():
    application = QApplication([])
    setStyle(application)
    mw = Window()
    mw.show()
    application.exec_()