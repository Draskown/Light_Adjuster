import cv2, numpy as np, random

from keras_preprocessing.image import ImageDataGenerator
from os import mkdir, walk, remove as remove_file
from os.path import join as join_paths, \
    exists as path_exists, basename
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D,\
    Flatten, Dense, Dropout
from json_loader import JsonHandler

# Глобальные переменные настройки
# Для обучения нейронной сети
BATCH = 20
DROPOUT = 0.5
EPOCHS = 100
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
        self.json_handler = JsonHandler("Custom NN Images")
        self.train()

    # Подготавливает тренировочный и валидационные датасеты
    def prepare_dataset(self) -> None:
        # Проверка на то, был ли датасет прежде сформирован
        # И если был - выйти из метода
        if self.json_handler.get_dataset_state() == 1:
            return
        
        # Для каждого из найденных категорий
        # Сделать отдельную папку в каталоге для тестовых и валидационных изображений
        # Если они не существуют
        for label in self.json_handler.labels:
            path = join_paths(self.json_handler.dirs.test_dir, label)
            if not path_exists(path):
                mkdir(path)
            
            path = join_paths(self.json_handler.dirs.val_dir, label)
            if not path_exists(path):
                mkdir(path)

        # Инициализация каскадов Хаара для нахождения лиц в анфас и профиль
        front_face_cascade = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")
        profile_face_cascade = cv2.CascadeClassifier("Cascades/haarcascade_profileface.xml")

        # Поиск всех изображений в файле проекта
        for root, _, files in walk(self.json_handler.dirs.images_dir):
            # Инициализация индексов для переименования изображений
            index_test = index_train = index_val = common_index = 0
            
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
                    file_path = join_paths(self.json_handler.dirs.train_dir, label)
                    # Читает изображение из пути к файлу
                    img = cv2.imread(join_paths(file_path, file))

                    # Коэффициент для уменьшения размера изображения
                    resize_mult = img.shape[1] / TARGET_IMG_WIDTH
                    
                    # Уменьшает изображение на заданный коэффициент
                    img = cv2.resize(img,
                                (int(img.shape[1] / resize_mult), int(img.shape[0] / resize_mult)),
                                interpolation=cv2.INTER_AREA)

                    # Нахождение анфасов лиц на изображении
                    fronts = front_face_cascade.detectMultiScale(
                        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                        scaleFactor=SCALE_FACTOR,
                        minNeighbors=MIN_NEIGHBOURS
                    )

                    # Нахождение профилей лиц на изображении
                    profiles = profile_face_cascade.detectMultiScale(
                        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                        scaleFactor=SCALE_FACTOR,
                        minNeighbors=MIN_NEIGHBOURS
                    )

                    # Вызов функции нахождения самого ближнего лица
                    face = []
                    # Если каскад Хаара обнаружил анфас
                    if len(fronts) > 0:
                        face = self.find_face(fronts, img)
                    # Или если каскад Хаара обнаружил профиль лица
                    elif len(profiles) > 0:
                        face = self.find_face(profiles, img)

                    # Если не было найдено ни одного лица
                    if len(face) == 0:
                        # Удалить невалидный файл
                        remove_file(join_paths(file_path, file))
                        # Продолжить итерирование по оставшимся файлам
                        continue
                    
                    # Если же лицо было найдено -
                    # Проверка на максимальное количество
                    # Тренировочных и тестовых изображений
                    # Если больше - начать с нуля
                    # Ели меньше - продолжить счёт
                    if common_index > MAX_IMAGES:
                        common_index = 0
                    else:
                        common_index += 1
                    
                    # Каждое четвёртое изображение из подготовленных
                    # Изображений загружается в каталог валидации
                    if common_index % 7 == 0:
                        self.split_dataset(face, self.json_handler.dirs.val_dir, label, index_val)
                        index_val += TWEAKED_IMAGES
                    # Каждое седьмое изображение из подготовленных
                    # Загружается в каталог теста
                    elif common_index % 10 == 0:
                        self.split_dataset(face, self.json_handler.dirs.test_dir, label, index_test)
                        index_test += TWEAKED_IMAGES
                    else:
                        self.split_dataset(face, self.json_handler.dirs.train_dir, label, index_train)
                        index_train += TWEAKED_IMAGES

                    # Удаляет обработанное полное изображение
                    remove_file(join_paths(file_path, file))

        # Задать, что датасет был создан
        self.json_handler.set_dataset_state(1)

    # Возвращает лицо самого большого размера
    # Чтобы в приоритете был человек ближе к камере
    def find_face(self, cascade: list, frame: cv2.Mat) -> cv2.Mat:
        # Инициализация наибольшего размера как первого лица
        max_area = (cascade[0, 0] + cascade[0, 2]) *\
                    (cascade[0, 1] + cascade[0, 3])
        
        # Для каждого из найденных каскадом лиц
        for (x, y, w, h) in cascade:
            # Рассчитать его размер
            area = (x + w) * (y + h)
            # Если он больше изначального наибольшего -
            # Обновить наибольшее
            if area > max_area:
                max_area = area

        # Повторное итерирование через
        # Все найденные лица, чтобы вернуть
        # Лицо, ближайшее к камере
        for (x, y, w, h) in cascade:
            if (x + w) * (y + h) == max_area:
                return frame[y:y + h, x:x + w]


    # Применяет случайную обработку к изображению
    def tweak_image(self, img) -> cv2.Mat:
        # Добавляет шум к изображению
        noise = np.zeros(img.shape, np.int16)
        cv2.randn(noise, 0, 30)
        img = cv2.add(img, noise, dtype=cv2.CV_8UC3)

        # Поворачивает изображение
        angle = random.randint(-3, 3)
        rows, cols = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
        img = cv2.warpAffine(img, M, (cols, rows))

        # Осветляет или затемняет изображение
        img_float = np.float32(img)
        brightness_value = np.random.uniform(-0.2, 0.2) * 255.0
        img_float += brightness_value
        img_float = np.clip(img_float, 0, 255)
        img = np.uint8(img_float)

        # Переворачивает изображение по горизонтали
        if random.randint(0, 1) == 1:
            img = cv2.flip(img, 1)

        # Возвращает изменённое изображение
        return img

    # Разделяет датасет на поданые каталоги
    # И добавляет обработанные дупликаты
    def split_dataset(self, face, destination, label, index):
        # Новое расположение файла в каталоге
        new_file_path = join_paths(destination, label)
        # Для каждого тренировочного файла должно быть
        # Сделано ещё пять таких же с применённой обработкой
        for _ in range (0, TWEAKED_IMAGES):
            # Новое имя файла = индекс изображений
            new_file = str(index) + ".jpg"
            # Применяет обработку к изображению
            tweaked = self.tweak_image(face)
            # Запись изображения найденного лица как файл
            # С новым именем в новом каталоге
            cv2.imwrite(join_paths(new_file_path, new_file), tweaked)
            # Инкремент индекса изображения
            index += 1

    # Обучает нейронную сеть
    def train(self):
        # Подготавливает датасет для тренировки нейронной сети
        # И получает от метода необходимые
        # Названия категорий и каталогов из файла json
        self.prepare_dataset()

        # Если модель обучена хорошо
        # То можно сразу перейти к детектированию
        if self.json_handler.get_model_state() == 1:
            self.detect_person()

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
            Dense(len(self.json_handler.labels), activation="softmax")
        ])

        # Компиляция параметров потери, оптимизации и вывода
        # Для нейронной сети
        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
        )

        # Инициализация генератора данных
        data_generator = ImageDataGenerator(rescale=1 / 255.0)

        # Генерация данных из тренировочного каталога
        train_data = data_generator.flow_from_directory(
            self.json_handler.dirs.train_dir,
            target_size=(64, 64),
            batch_size=BATCH,
            class_mode="categorical"
        )

        # Генерация данных из валидационного каталога
        val_data = data_generator.flow_from_directory(
            self.json_handler.dirs.val_dir,
            target_size=(64, 64),
            batch_size=BATCH,
            class_mode="categorical",
        )

        # Генерация данных из тестового каталога
        test_data = data_generator.flow_from_directory(
            self.json_handler.dirs.test_dir,
            target_size=(64, 64),
            batch_size=BATCH,
            class_mode="categorical",
        )

        # Тренировка нейронной сети
        model.fit(
            train_data,
            steps_per_epoch=len(train_data),
            epochs=EPOCHS,
            validation_data=val_data,
            validation_steps=len(test_data)
        )

        # Провести тест работы нейронной сети
        _, test_accuracy = model.evaluate(test_data, steps=len(test_data))

        # И если точность достаточно высокая
        # Можно считать, что модель дальше обучать
        # Не имеет смысла
        if test_accuracy > 0.85:
            self.json_handler.set_model_state(1)

        # Если папка Models не существует - создать её
        if not path_exists(self.json_handler.dirs.model_dir):
            mkdir(self.json_handler.dirs.model_dir)

        # Сохранить модель
        model.save(join_paths(self.json_handler.dirs.model_dir, "model.h5"))

        # Детектировать лица
        self.detect_person()

    # Детектирует лица с видеопотока и подаёт на вход 
    # Обученной модели для классификации
    def detect_person(self):
        # Загружает обученную модель
        model = load_model(join_paths(self.json_handler.dirs.model_dir, "model.h5"))

        # Преобразование словаря в один массив
        labels = {v: k for k, v in self.json_handler.labels.items()}
        
        # Инициализация массива частоты найденных категорий
        # Пока не используется
        # labelsFreq = []
        # for _ in labels:
        #     labelsFreq.append(0)

        # Инициализация анфаса лица каскада Хаара
        face_cascade = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")

        # Инициализация захвата видеопотока
        cap = cv2.VideoCapture(0)

        # Цикл обработки видеопотока
        while True:
            # Считывает один кадр с видеопотока
            ret, frame = cap.read()

            if not ret:
                continue

            # Детектирует лицо на кадре с ипользованием каскада
            faces_front = face_cascade.detectMultiScale(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                scaleFactor=SCALE_FACTOR,
                minNeighbors=MIN_NEIGHBOURS
            )

            # Цикл итерации каждого из найденных лиц
            for (x, y, w, h) in faces_front:
                # Рисует прямоугольник вокруг найденного лица
                frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Создаёт один массив из кадра
                img_array = frame[y:y + h, x:x + w]

                # Нормализирует массив
                new_array = cv2.resize(img_array, (64, 64)) / 255.0
                new_array = np.expand_dims(new_array, axis=0)

                # Использует модель для предположения
                # К какой категории принадлежит найденное лицо
                result_list = model.predict([new_array])
                
                # Выводит категорию с наибольшим процентом уверенности
                result = np.argmax((result_list[0]))

                # Выводит на изображение текст предполагаемой категории
                frame = cv2.putText(frame, 
                                    list(self.json_handler.labels.keys())[result], 
                                    (50, 50), 
                                    cv2.FONT_HERSHEY_COMPLEX, 
                                    1, (0, 255, 0), 2, cv2.LINE_AA)

                # Выводит на изображение текст уверенности
                # Предполагаемой категории
                frame = cv2.putText(frame, 
                                    str(result_list[0][result]), 
                                    (400, 50), 
                                    cv2.FONT_HERSHEY_COMPLEX, 
                                    1, (0, 255, 0), 2, cv2.LINE_AA)

                # labelsFreq[result] += 1

            # Показывает обработанный кадр
            cv2.imshow("frame", frame)

            # Выходит из цикла, если на окне
            # Бал нажата клавиша q
            if cv2.waitKey(20) and 0xFF == ord('q'):
                break
        
        # Закрывает видеопоток
        cap.release()
        cv2.destroyAllWindows()

        # Планируется имплементировать в будущем
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
