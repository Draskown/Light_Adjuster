from keras_preprocessing.image import ImageDataGenerator
from os import mkdir
from os.path import join as pathJoin, \
    exists as pathExists
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,\
    Flatten, Dense, Dropout
from loadJson import loadDirs

BATCH = 5
DROPOUT = 0.5
EPOCHS = 5


def train():
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

    if not pathExists(directories["model"]):
        mkdir(directories["model"])

    model.save(pathJoin(directories["model"], "model1.h5"))
