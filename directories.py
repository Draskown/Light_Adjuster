from os.path import join as join_paths, dirname, \
    abspath

# Класс для хранения каталогов
class Directories():
    # Инициализация нужных для работы каталогов
    base_dir = dirname(abspath(__file__))
    model_dir = join_paths(base_dir, "Model")
    images_dir = join_paths(base_dir, "Custom NN Images")
    train_dir = join_paths(images_dir, "Train")
    val_dir = join_paths(images_dir, "Validation")
    test_dir = join_paths(images_dir, "Test")