from keras_preprocessing.image import ImageDataGenerator
from os import mkdir, walk, remove as removeFile
from os.path import join as pathJoin, \
    exists as pathExists, basename
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,\
    Flatten, Dense, Dropout
from loadJson import loadDirs
import cv2, numpy as np, random

BATCH = 5
DROPOUT = 0.5
EPOCHS = 5
SCALE_FACTOR = 1.2
MIN_NEIGHBOURS = 3

def prepareDataset():
    loadDirs()

    from loadJson import labels, directories

    for label in labels:
        path = pathJoin(directories["test"], label)
        if not pathExists(path):
            mkdir(path)

    frontFaceCascade = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")
    profileFaceCascade = cv2.CascadeClassifier("Cascades/haarcascade_profileface.xml")

    currentId = 0
    for root, _, files in walk(directories["images"]):
        
        indexTest = indexTrain = commonIndex = 0
        if "Images\\Test" in root:
            continue

        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                label = basename(root)
                
                filePath = pathJoin(directories["train"], label)
                img = cv2.imread(pathJoin(filePath, file))

                img = cv2.resize(img,
                            (int(img.shape[1] / 6), int(img.shape[0] / 6)),
                            interpolation=cv2.INTER_AREA)

                facesCascade = frontFaceCascade.detectMultiScale(
                    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                    scaleFactor=SCALE_FACTOR,
                    minNeighbors=MIN_NEIGHBOURS
                )

                profileCascade = profileFaceCascade.detectMultiScale(
                    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                    scaleFactor=SCALE_FACTOR,
                    minNeighbors=MIN_NEIGHBOURS
                )

                face = []
                if len(facesCascade) > 0:
                    face = findFace(facesCascade, img)
                elif len(profileCascade) > 0:
                    face = findFace(profileCascade, img)

                if len(face) == 0:
                    removeFile(pathJoin(filePath, file))
                    continue

                if commonIndex > 250:
                    commonIndex = 0
                else:
                    commonIndex += 1

                if commonIndex % 4 == 0:
                    newFile = str(indexTest) + ".jpg"
                    newFilePath = pathJoin(directories["test"], label)
                    cv2.imwrite(pathJoin(newFilePath, newFile), face)
                    indexTest += 1
                else:
                    newFilePath = pathJoin(directories["train"], label)
                    for _ in range (0, 3):
                        newFile = str(indexTrain) + ".jpg"
                        
                        face = tweakImage(face)
                        cv2.imwrite(pathJoin(newFilePath, newFile), face)
                        
                        indexTrain += 1

                removeFile(pathJoin(filePath, file))

    return labels, directories

def findFace(cascade, frame):
    maxArea = (cascade[0, 0] + cascade[0, 2]) *\
                (cascade[0, 1] + cascade[0, 3])
    
    for (x, y, w, h) in cascade:
        area = (x + w) * (y + h)
        if area > maxArea:
            maxArea = area

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

def train():
    labels, directories = prepareDataset()

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

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    dataGen = ImageDataGenerator(rescale=1 / 255.0)

    trainData = dataGen.flow_from_directory(
        directories["train"],
        target_size=(64, 64),
        batch_size=BATCH,
        class_mode="categorical"
    )

    testData = dataGen.flow_from_directory(
        directories["test"],
        target_size=(64, 64),
        batch_size=BATCH,
        class_mode="categorical",
    )

    model.fit(
        trainData,
        validation_data=testData,
        epochs=5,
        steps_per_epoch=len(trainData),
        validation_steps=len(testData)
    )

    if not pathExists(directories["model"]):
        mkdir(directories["model"])

    model.save(pathJoin(directories["model"], "model.h5"))
