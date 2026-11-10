from helpers import *
import numpy as np


def main():
    data_path = "dataset"

    x_train = np.load("x_train_preprocessed.npy")
    x_test = np.load("x_test_preprocessed.npy")
    y_train = np.load("y_train_preprocessed.npy")
    train_ids = np.load("train_ids.npy")
    test_ids = np.load("test_ids.npy")

    

    submission_path = "submission.csv"
    create_csv_submission(test_ids, y_pred, submission_path)

if __name__ == "__main__":
    main()

