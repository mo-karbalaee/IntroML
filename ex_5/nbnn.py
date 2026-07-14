import numpy as np


class NBNNClassifier:
    def __init__(self, metric="euclidean"):
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
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must have shape (n_samples, n_features).")
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")

        self.X_train = X
        self.y_train = y
        """
        The unique list of items in the label array is all
        the possible labels. Easy. This method is like KNN's fit 
        method since it is also a lazy learner. 
        """
        self.classes_ = np.unique(y)
        return self

    def _euclidean_distances(self, x):
        """Return the Euclidean distance from x to all training samples."""
        """
        The same as KNN. 
        """

        return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

    def _cosine_distances(self, x):
        """
        Return the cosine distance from x to all training samples.

        Use the same convention as in knn.py:
            cosine_distance = 1 - cosine_similarity
        """
        
        """
        The same as KNN. 
        """

        x_norm = np.linalg.norm(x)
        train_norms = np.linalg.norm(self.X_train, axis=1)
        denominator = train_norms * x_norm

        dot_products = self.X_train @ x
        similarity = np.divide(
            dot_products,
            denominator,
            out=np.zeros_like(dot_products, dtype=np.float64),
            where=denominator != 0,
        )
        return 1.0 - similarity

    def _class_scores(self, distances):
        """
        Compute one score per class.

        For each class, use the distance of the nearest training sample from
        that class. The predicted class is the class with the smallest score.
        
        Nothing new here. It is just the simple implementation of the docstring. 
        """

        return np.array([distances[self.y_train == cls].min() for cls in self.classes_])

    def predict(self, X):
        """
        Predict labels for one or more samples with the NBNN rule.

        Requirements:
            - allow either a single sample or a batch
            - compute distances to all training samples
            - convert them into class-wise scores
            - return the class label with the smallest score
        """
        X = np.asarray(X, dtype=np.float64)

        if self.X_train is None or self.y_train is None:
            raise ValueError("The classifier must be fitted before calling predict().")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictions = []
        for x in X:
            if self.metric == "euclidean":
                distances = self._euclidean_distances(x)
            elif self.metric == "cosine":
                distances = self._cosine_distances(x)
            else:
                raise ValueError(f"Unsupported metric: {self.metric}")

            scores = self._class_scores(distances)
            predictions.append(self.classes_[np.argmin(scores)])

        return np.asarray(predictions)
