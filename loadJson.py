from os.path import join as pathJoin, dirname, \
    abspath, basename
from os import walk
from json import load, dump

# Словари для хранения используемых директорий и миён пользователей
labels = {}
directories = {}


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


def dump_parameters(params=None):
    if "model" not in directories or \
            params is None:
        params = [0, 0, 0]

    v = list(map(int, params))

    with open(pathJoin(directories["model"], "mainInfo.json"), "r") as f:
        a = load(f)
        a["parameters"] = {"left": v[0], "right": v[1], "seat": v[2]}

    with open(pathJoin(directories["model"], "mainInfo.json"), "w") as f:
        dump(a, f, ensure_ascii=False, indent=4)


def pull_parameters():
    if "model" in directories:
        with open(pathJoin(directories["model"], "mainInfo.json"), "r") as f:
            return list(load(f)["parameters"].values())
    else:
        return [0, 0, 0]
