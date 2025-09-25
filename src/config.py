# src/config.py
import torch
import os

# --- Project Paths ---
DATA_DIR = r"C:\Users\dhruv\Desktop\640x360\640x360"
TRAIN_IMG_DIR = f"{DATA_DIR}\\train\\images\\"
TRAIN_MASK_DIR = f"{DATA_DIR}\\train\\labels\\areas\\train_labels_1920x1080\\"
TRAIN_LINE_JSON = f"{DATA_DIR}\\train\\labels\\lines\\train_labels_640x360.json"

TEST_IMG_DIR = f"{DATA_DIR}\\test\\images\\"
TEST_MASK_DIR = f"{DATA_DIR}\\test\\labels\\areas\\test_labels_1920x1080\\"  # For evaluation
TEST_LINE_JSON = f"{DATA_DIR}\\test\\labels\\lines\\test_labels_640x360.json"

# --- Output Paths (absolute) ---
PROJECT_ROOT = r"C:\Users\dhruv\ackops\runway-segmentation-hackathon"
MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")

# Make sure folders exist
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

# --- Model Hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 16  # Adjust based on your GPU memory (8-16 recommended)
NUM_EPOCHS = 75  # (75-100 recommended)
NUM_WORKERS = 2  # For DataLoader
IMAGE_HEIGHT = 360  # For 640x360 resolution
IMAGE_WIDTH = 640
PIN_MEMORY = True

# --- Training & Preprocessing ---
LOAD_MODEL = False  # Set to True to load a pre-trained model
PRETRAINED_ENCODER = 'resnet34'  # Encoder for U-Net
ENCODER_WEIGHTS = 'imagenet'
NUM_CLASSES = 2  # 1 for runway + 1 for background
NUM_LINE_COORDS = 4 * 3  # 12 coordinates (3 lines with start & end points)

# --- Early Stopping ---
EARLY_STOPPING_PATIENCE = 20  # Epochs to wait for improvement
EARLY_STOPPING_DELTA = 0.001  # Minimum change to count as improvement

# --- Evaluation ---
CONFIDENCE_THRESHOLD = 0.5  # Threshold for binary mask conversion
