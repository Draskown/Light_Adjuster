from os.path import join as pathJoin, dirname, \
    abspath

# Класс для хранения каталогов
class Directories():
    # Нахождение нужных для работы каталогов
    baseDir = dirname(abspath(__file__))
    modelDir = pathJoin(baseDir, "Model")
    imagesDir = pathJoin(baseDir, "Images")
    trainDir = pathJoin(imagesDir, "Train")
    valDir = pathJoin(imagesDir, "Validation")
    testDir = pathJoin(imagesDir, "Test")