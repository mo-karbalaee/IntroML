import numpy as np


def confusion_matrix(y_true, y_pred):
    """
    Compute a confusion matrix using NumPy only.

    Requirements:
        - convert both inputs to NumPy arrays
        - check that they have the same one-dimensional shape
        - infer the number of classes from the largest observed label
        - count how often every pair (true_label, pred_label) occurs
        - return an integer matrix of shape (num_classes, num_classes)
    """
    # Convert inputs to NumPy arrays.
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Validate the shapes.

    # Handle the empty-input case.

    # Infer the number of classes.

    # Initialize the confusion matrix.

    # Fill the confusion matrix.

    return cm
