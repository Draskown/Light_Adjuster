import cv2, numpy as np, random

from keras_preprocessing.image import ImageDataGenerator
from os import mkdir, walk, remove as removeFile
from os.path import join as pathJoin, \
    exists as pathExists, basename
from keras import losses
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D,\
    Flatten, Dense, Dropout
from loadJson import JsonHandler

# Глобальные переменные настройки
# Для обучения нейронной сети
BATCH = 5
DROPOUT = 0.5
EPOCHS = 15
# Для нахождения лиц
SCALE_FACTOR = 1.2
MIN_NEIGHBOURS = 3
# Для ширины изображения, подающегося на вход каскадов Хаара
TARGET_IMG_WIDTH = 350
# Для максимального количества изображений датасета
MAX_IMAGES = 500
# Количество обработанных дополнительных изображений
TWEAKED_IMAGES = 5

# Класс, обрабатывающий детекцию лица
class FaceReconizer():
    def __init__(self) -> None:
        # Инициализация объекта класса
        # Обработчика json файла
        # И вызов метода, выполняющего обучение сети
        self.jh = JsonHandler()
        self.train()

    # Подготавливает тренировочный и валидационные датасеты
    def prepareDataset(self):
        # Проверка на то, был ли датасет прежде сформирован
        # И если был - выйти из метода
        if self.jh.getDatasetState() == 1:
            return
        
        # Для каждого из найденных категорий
        # Сделать отдельную папку в каталоге для тестовых и валидационных изображений
        # Если они не существуют
        for label in self.jh.labels:
            path = pathJoin(self.jh.dirs.testDir, label)
            if not pathExists(path):
                mkdir(path)
            
            path = pathJoin(self.jh.dirs.valDir, label)
            if not pathExists(path):
                mkdir(path)

        # Инициализация каскадов Хаара для нахождения лиц в анфас и профиль
        frontFaceCascade = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")
        profileFaceCascade = cv2.CascadeClassifier("Cascades/haarcascade_profileface.xml")

        # Поиск всех изображений в файле проекта
        for root, _, files in walk(self.jh.dirs.imagesDir):
            # Инициализация индексов для переименования изображений
            indexTest = indexTrain = indexVal = commonIndex = 0
            
            # Пропустить каталог с тестовыми и валидационными изображениями
            if "Images\\Test" in root or "Images\\Validation" in root:
                continue
            
            # Цикл всех файлов
            for file in files:
                # Если файл оканчивается на расширение изображения
                if file.endswith("png") or file.endswith("jpg"):
                    # Отдельное поле для хранения категории
                    label = basename(root)
                    
                    # Создаёт путь к файлу из папки тренировочных изображений
                    # И названия категории
                    filePath = pathJoin(self.jh.dirs.trainDir, label)
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
                        face = self.findFace(facesCascade, img)
                    # Или если каскад Хаара обнаружил профиль лица
                    elif len(profileCascade) > 0:
                        face = self.findFace(profileCascade, img)

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
                        self.splitDataset(face, self.jh.dirs.valDir, label, indexVal)
                        indexVal += TWEAKED_IMAGES
                    # Каждое седьмое изображение из подготовленных
                    # Загружается в каталог теста
                    elif commonIndex % 7 == 0:
                        self.splitDataset(face, self.jh.dirs.testDir, label, indexTest)
                        indexTest += TWEAKED_IMAGES
                    else:
                        self.splitDataset(face, self.jh.dirs.trainDir, label, indexTest)
                        indexTrain += TWEAKED_IMAGES

                    # Удаляет обработанное полное изображение
                    removeFile(pathJoin(filePath, file))

        # Задать, что датасет был создан
        self.jh.setDatasetState(1)

    # Возвращает лицо самого большого размера
    # Чтобы в приоритете был человек ближе к камере
    def findFace(self, cascade, frame):
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
    def tweakImage(self, img):
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

    # Разделяет датасет на поданые каталоги
    # И добавляет обработанные дупликаты
    def splitDataset(self, face, destination, label, index):
        # Новое расположение файла в каталоге
        newFilePath = pathJoin(destination, label)
        # Для каждого тренировочного файла должно быть
        # Сделано ещё пять таких же с применённой обработкой
        for _ in range (0, TWEAKED_IMAGES):
            # Новое имя файла = индекс изображений
            newFile = str(index) + ".jpg"
            # Применяет обработку к изображению
            tweaked = self.tweakImage(face)
            # Запись изображения найденного лица как файл
            # С новым именем в новом каталоге
            cv2.imwrite(pathJoin(newFilePath, newFile), tweaked)
            # Инкремент индекса изображения
            index += 1

    # Обучает нейронную сеть
    def train(self):
        # Подготавливает датасет для тренировки нейронной сети
        # И получает от метода необходимые
        # Названия категорий и каталогов из файла json
        self.prepareDataset()

        # Если модель обучена хорошо
        # То можно сразу перейти к детектированию
        if self.jh.getModelState() == 1:
            self.detectPerson()

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
            Dense(len(self.jh.labels), activation="softmax")
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
            self.jh.dirs.trainDir,
            target_size=(64, 64),
            batch_size=BATCH,
            class_mode="categorical"
        )

        # Генерация данных из валидационного каталога
        valData = dataGen.flow_from_directory(
            self.jh.dirs.valDir,
            target_size=(64, 64),
            batch_size=BATCH,
            class_mode="categorical",
        )

        # Генерация данных из тестового каталога
        testData = dataGen.flow_from_directory(
            self.jh.dirs.testDir,
            target_size=(64, 64),
            batch_size=BATCH,
            class_mode="categorical",
        )

        # Тренировка нейронной сети
        model.fit(
            trainData,
            steps_per_epoch=len(trainData),
            epochs=EPOCHS,
            validation_data=valData,
            validation_steps=len(testData)
        )

        # Провести тест работы нейронной сети
        _, testAcc = model.evaluate(testData, steps=len(testData))

        # И если точность достаточно высокая
        # Можно считать, что модель дальше обучать
        # Не имеет смысла
        if testAcc > 0.85:
            self.jh.setModelState(1)

        # Если папка Models не существует - создать её
        if not pathExists(self.jh.dirs.modelDir):
            mkdir(self.jh.dirs.modelDir)

        # Сохранить модель
        model.save(pathJoin(self.jh.dirs.modelDir, "model.h5"))

        # Детектировать лица
        self.detectPerson()

    # Детектирует лица с видеопотока и подаёт на вход 
    # Обученной модели для классификации
    def detectPerson(self):
        model = load_model(pathJoin(self.jh.dirs.modelDir, "model.h5"))

        labels = {v: k for k, v in self.jh.labels.items()}
        labelsFreq = []
        for _ in labels:
            labelsFreq.append(0)

        faceCascade = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")

        cap = cv2.VideoCapture(0)

        while True:
            _, frame = cap.read()

            faces_front = faceCascade.detectMultiScale(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                scaleFactor=1.5,
                minNeighbors=3
            )

            for (x, y, w, h) in faces_front:
                frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                imgArray = frame[y:y + h, x:x + w]

                newArray = cv2.resize(imgArray, (64, 64)) / 255.0
                newArray = np.expand_dims(newArray, axis=0)

                resultList = model.predict([newArray])
                
                result = np.argmax((resultList[0]))

                frame = cv2.putText(frame, 
                                    str(result), 
                                    (50, 50), 
                                    cv2.FONT_HERSHEY_COMPLEX, 
                                    1, (0, 255, 0), 2, cv2.LINE_AA)

                frame = cv2.putText(frame, 
                                    str(resultList[0][result]), 
                                    (100, 50), 
                                    cv2.FONT_HERSHEY_COMPLEX, 
                                    1, (0, 255, 0), 2, cv2.LINE_AA)

                labelsFreq[result] += 1

            cv2.imshow("frame", frame)

            if cv2.waitKey(20) and 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # i = 0
        # while i < 25:
        #     ret, frame = cap.read()

        #     faces = faceCascade.detectMultiScale(
        #         cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
        #         scaleFactor=SCALE_FACTOR,
        #         minNeighbors=MIN_NEIGHBOURS
        #     )

        #     for (x, y, w, h) in faces:
        #         imgArray = frame[y:y + h, x:x + w]

        #         newArray = cv2.resize(imgArray, (64, 64)) / 255.0
        #         newArray = cv2.expand_dims(newArray, axis=0)

        #         result_1 = model.predict([newArray])
        #         result = cv2.argmax((result_1[0]))

        #         labelsFreq[result] += 1

        #     i += 1

        #     print(labelsFreq)

        #     if cv2.waitKey(20) and 0xFF == ord('q'):
        #         break

        # cap.release()
        # cv2.destroyAllWindows()

        # print(labelsFreq)

        # return labels[labelsFreq.index(max(labelsFreq))]