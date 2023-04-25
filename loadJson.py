from os.path import join as pathJoin, dirname, \
    abspath, basename
from os import walk
from json import load, dump

# Словари для хранения используемых директорий и миён пользователей
labels = {}
directories = {}

# Загружает найденные каталоги в json файл
def loadDirs():
    global labels, directories

    # Нахождение нужных для работы каталогов
    baseDir = dirname(abspath(__file__))
    modelDir = pathJoin(baseDir, "Model")
    imageDir = pathJoin(baseDir, "Images")
    trainDir = pathJoin(imageDir, "Train")
    testDir = pathJoin(imageDir, "Test")

    # Открывыает главный json файл, где хранятся нужные сведения
    with open(pathJoin(baseDir, "mainInfo.json"), "r") as f:
        a = load(f)

    # Задаёт словарь директорий
    directories = {"base": baseDir,
                   "model": modelDir,
                   "images": imageDir,
                   "train": trainDir,
                   "test": testDir}

    # Задаёт идентификатор для каждого имени
    # Начиная с нулевого
    currentId = 0
    for root, _, files in walk(directories["images"]):
        for file in files:
            # Каждый файл проверяется на то, является ли он изображением
            if file.endswith("png") or file.endswith("jpg"):
                # Узнаёт имя папки, в который файл лежит
                label = basename(root)
                # Если такого имени нет - добавить его в список
                if label not in labels:
                    labels[label] = currentId
                    currentId += 1

    # Определяет json файл с созданными словарями
    a["directories"] = directories
    a["labels"] = labels

    # Обновляет json файл
    with open(pathJoin(baseDir, "mainInfo.json"), "w") as f:
        dump(a, f, ensure_ascii=False, indent=4)

# Возвращает состояние модели нейронной сети
def getModelState():
    # Открывает json файл
    with open(pathJoin(directories["base"], "mainInfo.json"), "r") as f:
        # И возвращает нужное значение
        return load(f)["model_is_ok"]

# Устанавливает состояние модели нейронной сети
def setModelState(state):
    # Открывает файл для чтения
    with open(pathJoin(directories["base"], "mainInfo.json"), "r") as f:
        # Записывает в локальный словарь необходимое значение
        # Состояния модели
        a = load(f)
        a["model_is_ok"] = state

    # Открывает файл снова и перезаписывает json файл
    with open(pathJoin(directories["base"], "mainInfo.json"), "w") as f:
        dump(a, f, ensure_ascii=False, indent=4)

def dumpParameters(params=None):
    if "model" not in directories or \
            params is None:
        params = [0, 0, 0]

    v = list(map(int, params))

    with open(pathJoin(directories["base"], "mainInfo.json"), "r") as f:
        a = load(f)
        a["parameters"] = {"left": v[0], "right": v[1], "seat": v[2]}

    with open(pathJoin(directories["base"], "mainInfo.json"), "w") as f:
        dump(a, f, ensure_ascii=False, indent=4)


def pullParameters():
    if "model" in directories:
        with open(pathJoin(directories["base"], "mainInfo.json"), "r") as f:
            return list(load(f)["parameters"].values())
    else:
        return [0, 0, 0]
