import numpy as np


def confusion_matrix(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if y_true.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional.")

    if y_true.size == 0:
        return np.zeros((0, 0), dtype=int)

    num_classes = int(max(y_true.max(), y_pred.max())) + 1

    cm = np.zeros((num_classes, num_classes), dtype=int)

    np.add.at(cm, (y_true.astype(int), y_pred.astype(int)), 1)

    return cm
