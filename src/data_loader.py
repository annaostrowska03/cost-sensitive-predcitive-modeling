import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_project_data(data_dir="..\\data"):
    """
    Loads the train and test files used in the project from the given directory.
    """
    x_train_path = os.path.join(data_dir, "x_train.txt")
    y_train_path = os.path.join(data_dir, "y_train.txt")
    x_test_path = os.path.join(data_dir, "x_test.txt")
    
    X_train = pd.read_csv(x_train_path, sep='\\s+')
    y_train = pd.read_csv(y_train_path, sep='\\s+')
    X_test = pd.read_csv(x_test_path, sep='\\s+')
    logger.info(f"Data loaded. X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}")
    
    return X_train, y_train, X_test
