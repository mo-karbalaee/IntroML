import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
except ImportError:
    LogisticRegression = None


"""
We are just developing a wrapper here. The mathematical heavy lifting is 
done by sklearn. This wrapper is designed to do some validations and give
the structure that the main file can treat this classifier the same way it 
treats the other two. Have in mind that logistic regression is an eager learner.
Not a lazy learner like KNN and NBNN.
"""
class LogisticRegressionClassifier:
    def __init__(self, max_iter=2000, random_state=0):
        """
        Ok what are these two? logistic regression is an iterative algorithm, meaning, 
        it will not find the optimal weights one shot. We can specify how many iterations
        we want it to continue the optimization. 
        The random state is a little trickier but nothing so hard to understand. 
        There are some random operations in the algorithm that might results into 
        slightly different final weights. Since we want to use the model in a stable 
        fashion for inference, we set a seed for this random operations so that we can 
        use the same seed for inference. This way each inference is not using a slightly
        different model. 
        """
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None
        """
        We store it for validation reasons. 
        The underscore in the end is a sklearn syntax and has a meaning. 
        It indicates that this value will be set during training not by the user. 
        """
        self.n_features_ = None

    def fit(self, X, y):
        """
        Train a logistic-regression classifier on the given feature matrix.

        Requirements:
            - convert X and y to NumPy arrays
            - validate that X has shape (n_samples, n_features)
            - validate that y is one-dimensional
            - validate that X and y contain the same number of samples
            - create and fit sklearn.linear_model.LogisticRegression
            - return self
        """
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
        """
        We return self because function chaining is cool. Remember?
        """
        return self

    def predict(self, X):
        """
        Predict labels for one or more input samples.

        Requirements:
            - raise an error if fit() was not called first
            - accept either a single sample or a full batch
            - validate the feature dimension
            - return the model predictions as a NumPy array
        """
        if self.model is None:
            raise ValueError("The classifier must be fitted before calling predict().")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != self.n_features_:
            raise ValueError(f"Expected {self.n_features_} features, got {X.shape[1]}.")

        """
        Nothing fancy here. Just run the predict on the model and ensure it is a 
        numpy array and return it. Easy and intuitive. 
        """
        return np.asarray(self.model.predict(X))
