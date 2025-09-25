FS2020 Runway Detection and Analysis
This project uses a U-Net model with a pre-trained ResNet encoder to perform runway segmentation and identify key runway lines (Left Edge, Right Edge, Center Line) from images in the FS2020 dataset.

Project Structure
data/: Contains the training and testing datasets. This directory should be populated with the FS2020 dataset as described in file_structure.md.

src/: Contains all the Python source code.

config.py: Main configuration file for paths, hyperparameters, etc.

dataset.py: Defines the data loading and preprocessing pipeline.

model.py: Defines the U-Net architecture with a multi-task head.

loss.py: Defines the combined loss function for segmentation and regression.

train.py: The main script to start the training process.

predict.py: Script to run inference on the test set and generate the submission.csv.

evaluate.py: Contains functions for calculating the required evaluation metrics.

models/: Trained model checkpoints will be saved here.

results/: The output submission.csv and any predicted masks will be saved here.

requirements.txt: A list of all Python dependencies.

Setup
Clone the repository and create the directory structure.

Download the Data:

Download the FS2020 Runway Dataset.

Organize the data into the data/ directory according to the structure outlined in file_structure.md. You will need to create the images and labels/areas subdirectories and move the corresponding files into them.

Install Dependencies:
It is recommended to use a virtual environment.

pip install -r requirements.txt

Note: segmentation-models-pytorch requires a compatible version of timm. If you encounter issues, you may need to install it manually: pip install timm.

How to Run
1. Training the Model
To start training the model, navigate to the src/ directory and run the train.py script.

cd src
python train.py

The script will use the parameters defined in config.py.

The best performing model checkpoint (best_model.pth) will be saved in the models/ directory.

The training process includes early stopping to prevent overfitting.

2. Generating Predictions and Evaluating
After training is complete, run the predict.py script from the src/ directory to generate predictions on the test set and create the final submission file.

cd src
python predict.py

This script will load the best_model.pth from the models/ directory.

It will process all images in the data/test/images directory.

It will calculate the IoU, Anchor, and Boolean scores for each test image.

A submission.csv file will be created in the results/ directory with the scores for each image and the overall mean scores.