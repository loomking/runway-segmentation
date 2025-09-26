import torch
import os

# Project Paths
DATA_DIR = r"C:\Users\dhruv\Desktop\640x360\640x360"
PROJECT_ROOT = r"C:\Users\dhruv\ackops\runway-segmentation-hackathon"

# Training Data
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train", "images")
TRAIN_MASK_DIR = os.path.join(DATA_DIR, "train", "labels", "areas", "train_labels_1920x1080")
TRAIN_LINE_JSON = os.path.join(DATA_DIR, "train", "labels", "lines", "train_labels_640x360.json")

# Test Data
TEST_IMG_DIR = os.path.join(DATA_DIR, "test", "images")
TEST_MASK_DIR = os.path.join(DATA_DIR, "test", "labels", "areas", "test_labels_1920x1080")
TEST_LINE_JSON = os.path.join(DATA_DIR, "test", "labels", "lines", "test_labels_640x360.json")

# Validation Data
VAL_IMG_DIR = TEST_IMG_DIR
VAL_MASK_DIR = TEST_MASK_DIR
VAL_LINE_JSON = TEST_LINE_JSON

# Output Paths 
MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

# Model parameters 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 16 
NUM_EPOCHS = 75 
NUM_WORKERS = 2
IMAGE_HEIGHT = 360
IMAGE_WIDTH = 640
PIN_MEMORY = True

# Training & Preprocessing
LOAD_MODEL = False
PRETRAINED_ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'

# Early Stopping 
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_DELTA = 0.001

# Evaluation
CONFIDENCE_THRESHOLD = 0.5
