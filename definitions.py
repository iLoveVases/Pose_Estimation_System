import os

CLASSES = ['standing', 'sitting', 'lying']  # THE ORDER OF ENTRIES MUST BE THE SAME

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Project Root

THUNDER_PATH = os.path.join(ROOT_DIR, 'Models/lite-model_movenet_singlepose_thunder_3.tflite')
LIGHTNING_PATH = os.path.join(ROOT_DIR, 'Models/lite-model_movenet_singlepose_lightning_3.tflite')
XGB_MODEL_PATH = os.path.join(ROOT_DIR, 'Models/XGB_models/xgb_222.json')
CSV_PATH = os.path.join(ROOT_DIR, 'learning_data/CSV')
CSV_NAME = "data"


LEARNING_DATA_PATH = os.path.join(ROOT_DIR, 'learning_data')

# .npy
STANDING_PATH_NPY = os.path.join(ROOT_DIR, r'learning_data\standing_npy')
SITTING_PATH_NPY = os.path.join(ROOT_DIR, r'learning_data\sitting_npy')
LYING_PATH_NPY = os.path.join(ROOT_DIR, r'learning_data\lying_npy')
IMAGES_PATHS_NPY = [STANDING_PATH_NPY, SITTING_PATH_NPY, LYING_PATH_NPY]  # THE ORDER OF ENTRIES MUST BE THE SAME

# images
STANDING_PATH = os.path.join(ROOT_DIR, r'learning_data\standing')
SITTING_PATH = os.path.join(ROOT_DIR, r'learning_data\sitting')
LYING_PATH = os.path.join(ROOT_DIR, r'learning_data\lying')
IMAGES_PATHS = [STANDING_PATH, SITTING_PATH, LYING_PATH]  # THE ORDER OF ENTRIES MUST BE THE SAME

# screenshots
STANDING_PATH_SCR = os.path.join(ROOT_DIR, r'screens\standing')
SITTING_PATH_SCR = os.path.join(ROOT_DIR, r'screens\sitting')
LYING_PATH_SCR = os.path.join(ROOT_DIR, r'screens\lying')
IMAGES_PATHS_SCR = [STANDING_PATH_SCR, SITTING_PATH_SCR, LYING_PATH_SCR]  # THE ORDER OF ENTRIES MUST BE THE SAME

