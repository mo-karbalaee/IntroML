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

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if y_true.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional.")

    if y_true.size == 0:
        return np.zeros((0, 0), dtype=int)
    
    """
    So we know that a confusion matrix is a square matrix. Meaning, its column and rows
    are have the same length. In order to find that length, we need to know the number of 
    classes. It would be as simple as getting the max of y_true and add it with one, because
    labels are zero indexed, if we knew that y_pred will only output real class values
    not some random bullshit. For example we have 10 classes and the classifier can 
    still say that this observation belongs to class 100. Which does not exist. 
    In order to not reach out of bound exception, we will consider that too. 
    And why we don't just work with y_pred then? because it is not guaranteed
    that the classifier has used all the classes in the inference. What if in a 10 class
    problem it maps every input to class 0? then we are fucked. So we get the max of each 
    then the max of both and we add 1 to it. This will be safe.
    """
    num_classes = int(max(y_true.max(), y_pred.max())) + 1
    
    """
    We first initialize the confusion matrix with zero values. 
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)

    """
    You know the shape of a confusion matrix right? You used it in your bachelor's thesis
    and I am proud of you because of that. This line will iterate over the zero 
    initialized confusion matrix and adds a one whenever is sees a tuple. 
    What is that tuple? pred and true. It will run for the number of occurrences 
    of that tuple. So if we have it 10 times, it does it ten times. Resulting into 10 
    written in that cell. 
    """
    np.add.at(cm, (y_true.astype(int), y_pred.astype(int)), 1)

    return cm
