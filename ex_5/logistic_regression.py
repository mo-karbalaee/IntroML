import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
except ImportError:
    LogisticRegression = None


class LogisticRegressionClassifier:
    def __init__(self, max_iter=2000, random_state=0):
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None
        self.n_features_ = None

    def fit(self, X, y):
        if LogisticRegression is None:
            raise ImportError(
                "scikit-learn is required for LogisticRegressionClassifier."
            )

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must have shape (n_samples, n_features).")
        if y.ndim != 1:
            raise ValueError("y must be one-dimensional.")
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")

        self.model = LogisticRegression(
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.model.fit(X, y)
        self.n_features_ = X.shape[1]
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("The classifier must be fitted before calling predict().")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != self.n_features_:
            raise ValueError(f"Expected {self.n_features_} features, got {X.shape[1]}.")

        return np.asarray(self.model.predict(X))
