from os.path import join as join_paths, basename
from os import walk
from json import load, dump
from directories import Directories

# Класс обработчика json файла
class JsonHandler():
    def __init__(self) -> None:
        # Создание объекта класса, хранящего 
        # Нужные для программы каталоги
        self.dirs = Directories()
        # Создаёт пустой словарь для хранения категорий
        self.labels = {}

        # Загружает категории в json файд
        self.load_labels()
    
    # Загружает найденные каталоги категорий в json файл
    def load_labels(self) -> None:
        # Открывыает json файл, где хранятся нужные сведения
        with open(join_paths(self.dirs.base_dir, "mainInfo.json"), "r") as f:
            a = load(f)

        # Задаёт идентификатор для каждого имени
        # Начиная с нулевого
        current_id = 0
        for root, _, files in walk(self.dirs.images_dir):
            for file in files:
                # Каждый файл проверяется на то, является ли он изображением
                if file.endswith("png") or file.endswith("jpg"):
                    # Узнаёт имя папки, в который файл лежит
                    label = basename(root)
                    # Если такого имени нет - добавить его в список
                    if label not in self.labels:
                        self.labels[label] = current_id
                        current_id += 1

        # Определяет json файл для категорий
        a["labels"] = self.labels

        # Обновляет json файл
        with open(join_paths(self.dirs.base_dir, "mainInfo.json"), "w") as f:
            dump(a, f, ensure_ascii=False, indent=4)

    # Возвращает состояние модели нейронной сети
    def get_model_state(self) -> int:
        # Открывает json файл
        with open(join_paths(self.dirs.base_dir, "mainInfo.json"), "r") as f:
            # И возвращает нужное значение
            return load(f)["model_is_ok"]

    # Устанавливает состояние модели нейронной сети
    def set_model_state(self, state: int) -> None:
        # Открывает файл для чтения
        with open(join_paths(self.dirs.base_dir, "mainInfo.json"), "r") as f:
            # Записывает в локальный словарь необходимое значение
            # Состояния модели
            a = load(f)
            a["model_is_ok"] = state

        # Открывает файл снова и перезаписывает json файл
        with open(join_paths(self.dirs.base_dir, "mainInfo.json"), "w") as f:
            dump(a, f, ensure_ascii=False, indent=4)

    # Возвращает состояние датасета для нейронной сети
    def get_dataset_state(self) -> int:
        # Открывает json файл
        with open(join_paths(self.dirs.base_dir, "mainInfo.json"), "r") as f:
            # И возвращает нужное значение
            return load(f)["dataset_created"]

    # Устанавливает состояние датасета для нейронной сети
    def set_dataset_state(self, state: int) -> None:
        # Открывает файл для чтения
        with open(join_paths(self.dirs.base_dir, "mainInfo.json"), "r") as f:
            # Записывает в локальный словарь необходимое значение
            # Состояния модели
            a = load(f)
            a["dataset_created"] = state

        # Открывает файл снова и перезаписывает json файл
        with open(join_paths(self.dirs.base_dir, "mainInfo.json"), "w") as f:
            dump(a, f, ensure_ascii=False, indent=4)


    # Планируется имплементировать в будущем
    # def dump_parameters(self, params=None):
    #     if params is None:
    #         params = [0, 0, 0]

    #     v = list(map(int, params))

    #     with open(join_paths(self.dirs.baseDir, "mainInfo.json"), "r") as f:
    #         a = load(f)
    #         a["parameters"] = {"left": v[0], "right": v[1], "seat": v[2]}

    #     with open(join_paths(self.dirs.baseDir, "mainInfo.json"), "w") as f:
    #         dump(a, f, ensure_ascii=False, indent=4)


    # def pull_parameters(self):
    #     with open(join_paths(self.dirs.baseDir, "mainInfo.json"), "r") as f:
    #         return list(load(f)["parameters"].values())
