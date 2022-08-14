import os

CLASSES = ['sitting', 'standing', 'lying']

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

# images
STANDING_PATH = os.path.join(ROOT_DIR, r'learning_data\standing')
SITTING_PATH = os.path.join(ROOT_DIR, r'learning_data\sitting')
LYING_PATH = os.path.join(ROOT_DIR, r'learning_data\lying')
IMAGES_PATHS = [STANDING_PATH, SITTING_PATH, LYING_PATH]

