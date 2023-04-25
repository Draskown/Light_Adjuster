from keras_preprocessing.image import ImageDataGenerator
from os import mkdir, walk, remove as removeFile
from os.path import join as pathJoin, \
    exists as pathExists, basename
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,\
    Flatten, Dense, Dropout
from loadJson import loadDirs
from cv2 import CascadeClassifier, imread, imwrite

BATCH = 5
DROPOUT = 0.5
EPOCHS = 5

def prepareDataset():
    loadDirs()

    from loadJson import labels, directories

    faceCascade = CascadeClassifier("Cascades")

    indexTest = indexTrain = 0

    commonIndex = 1

    currentId = 0
    for root, dirs, files in walk(directories["images"]):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                label = basename(root).replace(" ", "-").lower()
                
                filePath = pathJoin(directories["train"], label)
                img = imread(pathJoin(filePath, file))

                if commonIndex > 250:
                    commonIndex = 0
                else:
                    commonIndex += 1

                if commonIndex % 4 == 0:
                    newFile = str(indexTest)
                    newFilePath = pathJoin(directories["test"], label)
                    indexTest += 1
                else:
                    newFile = str(indexTrain)
                    indexTrain += 1

                newFile += ".png"

                imwrite(pathJoin(filePath, newFile))

                # removeFile(pathJoin(filePath, file))

def train():
    prepareDataset()
    
    loadDirs()

    from loadJson import labels, directories

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

    if not pathExists(directories["Model"]):
        mkdir(directories["Model"])

    model.save(pathJoin(directories["model"], "model.h5"))
