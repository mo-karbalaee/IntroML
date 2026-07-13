import numpy as np


class NBNNClassifier:
    def __init__(self, metric="euclidean"):
        self.metric = metric
        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must have shape (n_samples, n_features).")
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")

        self.X_train = X
        self.y_train = y
        self.classes_ = np.unique(y)
        return self

    def _euclidean_distances(self, x):
        return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

    def _cosine_distances(self, x):
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
        return np.array([distances[self.y_train == cls].min() for cls in self.classes_])

    def predict(self, X):
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
