import numpy as np


class NBNNClassifier:
    def __init__(self, metric="euclidean"):
        # Distance metric: "euclidean" or "cosine".
        self.metric = metric
        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def fit(self, X, y):
        """
        Store training data and labels as NumPy arrays.

        Requirements:
            - convert X and y to NumPy arrays
            - validate shapes
            - store the sorted unique class labels in self.classes_
            - return self
        """
        pass

    def _euclidean_distances(self, x):
        """Return the Euclidean distance from x to all training samples."""
        pass

    def _cosine_distances(self, x):
        """
        Return the cosine distance from x to all training samples.

        Use the same convention as in knn.py:
            cosine_distance = 1 - cosine_similarity
        """
        pass

    def _class_scores(self, distances):
        """
        Compute one score per class.

        For each class, use the distance of the nearest training sample from
        that class. The predicted class is the class with the smallest score.
        """
        pass

    def predict(self, X):
        """
        Predict labels for one or more samples with the NBNN rule.

        Requirements:
            - allow either a single sample or a batch
            - compute distances to all training samples
            - convert them into class-wise scores
            - return the class label with the smallest score
        """
        pass
