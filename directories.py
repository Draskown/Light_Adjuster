from os.path import join as join_paths, dirname, \
    abspath

# Класс для хранения каталогов
class Directories():
    def __init__(self, type: str) -> None:
        # Инициализация нужных для работы каталогов
        self.base_dir = dirname(abspath(__file__))
        self.model_dir = join_paths(self.base_dir, "Model")
        self.images_dir = join_paths(self.base_dir, type)
        self.train_dir = join_paths(self.images_dir, "Train")
        self.val_dir = join_paths(self.images_dir, "Validation")
        self.test_dir = join_paths(self.images_dir, "Test")
        