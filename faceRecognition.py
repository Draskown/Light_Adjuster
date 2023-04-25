import cv2, numpy as np, random

from keras_preprocessing.image import ImageDataGenerator
from os import mkdir, walk, remove as removeFile
from os.path import join as pathJoin, \
    exists as pathExists, basename
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D,\
    Flatten, Dense, Dropout
from loadJson import loadDirs

# Глобальные переменные настройки
# Для обучения нейронной сети
BATCH = 5
DROPOUT = 0.5
EPOCHS = 5
# Для нахождения лиц
SCALE_FACTOR = 1.2
MIN_NEIGHBOURS = 3
# Для ширины изображения, подающегося на вход каскадов Хаара
TARGET_IMG_WIDTH = 350
# Для максимального количества изображений датасета
MAX_IMAGES = 500

# Подготавливает тренировочный и валидационные датасеты
def prepareDataset():
    # Загружает каталоги из файла json
    loadDirs()

    # Импортирует найденные категории и каталоги
    # Из json файла
    from loadJson import labels, directories

    # Для каждого из найденных категорий
    # Сделать отдельную папку в каталоге для тестовых изображений
    # Если они не существуют
    for label in labels:
        path = pathJoin(directories["test"], label)
        if not pathExists(path):
            mkdir(path)

    # Инициализация каскадов Хаара для нахождения лиц в анфас и профиль
    frontFaceCascade = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")
    profileFaceCascade = cv2.CascadeClassifier("Cascades/haarcascade_profileface.xml")

    # Поиск всех изобрежений в файле проекта
    currentId = 0
    for root, _, files in walk(directories["images"]):
        
        # Инициализация индексов для переименования изображений
        indexTest = indexTrain = commonIndex = 0
        
        # Пропустить каталог с тестовыми изображениями
        if "Images\\Test" in root:
            continue
        
        # Цикл всех файлов
        for file in files:
            # Если файл оканчивается на расширение изображения
            if file.endswith("png") or file.endswith("jpg"):
                # Отдельное поле для хранения категории
                label = basename(root)
                
                # Создаёт путь к файлу из папки тренировочных изображений
                # И названия категории
                filePath = pathJoin(directories["train"], label)
                # Читает изображение из пути к файлу
                img = cv2.imread(pathJoin(filePath, file))

                # Коэффициент для уменьшения размера изображения
                resizeMultiplier = img.shape[1] / TARGET_IMG_WIDTH
                
                # Уменьшает изображение на заданный коэффициент
                img = cv2.resize(img,
                            (int(img.shape[1] / resizeMultiplier), int(img.shape[0] / resizeMultiplier)),
                            interpolation=cv2.INTER_AREA)

                # Нахождение анфасов лиц на изображении
                facesCascade = frontFaceCascade.detectMultiScale(
                    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                    scaleFactor=SCALE_FACTOR,
                    minNeighbors=MIN_NEIGHBOURS
                )

                # Нахождение профилей лиц на изображении
                profileCascade = profileFaceCascade.detectMultiScale(
                    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                    scaleFactor=SCALE_FACTOR,
                    minNeighbors=MIN_NEIGHBOURS
                )

                # Вызов функции нахождения самого ближнего лица
                face = []
                # Если каскад Хаара обнаружил анфас
                if len(facesCascade) > 0:
                    face = findFace(facesCascade, img)
                # Или если каскад Хаара обнаружил профиль лица
                elif len(profileCascade) > 0:
                    face = findFace(profileCascade, img)

                # Если не было найдено ни одного лица
                if len(face) == 0:
                    # Удалить невалидный файл
                    removeFile(pathJoin(filePath, file))
                    # Продолжить итерирование по оставшимся файлам
                    continue
                
                # Если же лицо было найдено -
                # Проверка на максимальное количество
                # Тренировочных и тестовых изображений
                # Если больше - начать с нуля
                # Ели меньше - продолжить счёт
                if commonIndex > MAX_IMAGES:
                    commonIndex = 0
                else:
                    commonIndex += 1
                
                # Каждое четвёртое изображение из подготовленных
                # Изображений загружается в каталог валидации
                if commonIndex % 4 == 0:
                    # Новое имя файла = индекс валидационных изображений
                    newFile = str(indexTest) + ".jpg"
                    # Новое расположение файла в валидационном каталоге
                    newFilePath = pathJoin(directories["test"], label)
                    # Запись изображения найденного лица как файл
                    # С новым именем в новом каталоге
                    cv2.imwrite(pathJoin(newFilePath, newFile), face)
                    # Инкремент индекса валидационных изображений
                    indexTest += 1
                else:
                    # Новое расположение файла в тренировочном каталоге
                    newFilePath = pathJoin(directories["train"], label)
                    # Для каждого тренировочного файла должно быть
                    # Сделано ещё три таких же с применённой обработкой
                    for _ in range (0, 3):
                        # Новое имя файла = индекс тренировочных изображений
                        newFile = str(indexTrain) + ".jpg"
                        
                        # Применяет обработку к изображению
                        face = tweakImage(face)
                        # Запись изображения найденного лица как файл
                        # С новым именем в новом каталоге
                        cv2.imwrite(pathJoin(newFilePath, newFile), face)
                        
                        # Инкремент индекса тренировочных изображений
                        indexTrain += 1

                # Удаляет обработанное полное изображение
                removeFile(pathJoin(filePath, file))

    # Возвращает список категорий и каталогов из файла json
    # Для дальнейшего использования 
    return labels, directories

# Возвращает лицо самого большого размера
# Чтобы в приоритете был человек ближе к камере
def findFace(cascade, frame):
    # Инициализация наибольшего размера как первого лица
    maxArea = (cascade[0, 0] + cascade[0, 2]) *\
                (cascade[0, 1] + cascade[0, 3])
    
    # Для каждого из найденных каскадом лиц
    for (x, y, w, h) in cascade:
        # Рассчитать его размер
        area = (x + w) * (y + h)
        # Если он больше изначального наибольшего -
        # Обновить наибольшее
        if area > maxArea:
            maxArea = area

    # Повторное итерирование через
    # Все найденные лица, чтобы вернуть
    # Лицо, ближайшее к камере
    for (x, y, w, h) in cascade:
        if (x + w) * (y + h) == maxArea:
            return frame[y:y + h, x:x + w]


# Применяет случайную обработку к изображению
def tweakImage(img):
    # Добавляет шум к изображению
    noise = np.zeros(img.shape, np.int16)
    cv2.randn(noise, 0, 30)
    img = cv2.add(img, noise, dtype=cv2.CV_8UC3)

    # Поворачивает изображение
    angle = random.randint(-3, 3)
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
    img = cv2.warpAffine(img, M, (cols, rows))

    # Переворачивает изображение по горизонтали
    if random.randint(0, 1) == 1:
        img = cv2.flip(img, 1)

    # Возвращает изменённое изображение
    return img

# Обучает нейронную сеть
def train():
    # Подготавливает датасет для тренировки нейронной сети
    # И получает от метода необходимые
    # Названия категорий и каталогов из файла json
    labels, directories = prepareDataset()

    # Инициализация структуры нейронной сети
    model = Sequential([
        Conv2D(16, (3, 3), padding="same", activation="relu", input_shape=(64, 64, 3)),
        Dropout(DROPOUT),
        Conv2D(16, (3, 3), padding="same", activation="relu"),
        MaxPooling2D((2, 2)),
        #
        Conv2D(8, (3, 3), padding="same", activation="relu"),
        Conv2D(8, (3, 3), padding="same", activation="relu"),
        MaxPooling2D((2, 2)),
        #
        Flatten(),
        Dense(128, activation="relu"),
        Dense(len(labels), activation="softmax")
    ])

    # Компиляция параметров потери, оптимизации и вывода
    # Для нейронной сети
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    # Инициализация генератора данных
    dataGen = ImageDataGenerator(rescale=1 / 255.0)

    # Генерация данных из тренировочного каталога
    trainData = dataGen.flow_from_directory(
        directories["train"],
        target_size=(64, 64),
        batch_size=BATCH,
        class_mode="categorical"
    )

    # Генерация данных из валидационного каталога
    testData = dataGen.flow_from_directory(
        directories["test"],
        target_size=(64, 64),
        batch_size=BATCH,
        class_mode="categorical",
    )

    # Тренировка нейронной сети
    model.fit(
        trainData,
        validation_data=testData,
        epochs=5,
        steps_per_epoch=len(trainData),
        validation_steps=len(testData)
    )

    # Если папка Models не существует - создать её
    if not pathExists(directories["model"]):
        mkdir(directories["model"])

    # Сохранить модель
    model.save(pathJoin(directories["model"], "model.h5"))