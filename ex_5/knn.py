from pathlib import Path

import numpy as np

from ex_5.visualization import plot_knn_neighbors


class KNNClassifier:
    def __init__(
        self,
        n_neighbors=3,
        metric="euclidean",
        plot_neighbors=False,
        image_shape=None,
        plot_dir=None,
        plot_prefix="knn",
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.plot_neighbors = plot_neighbors
        self.image_shape = image_shape
        self.plot_dir = plot_dir
        self.plot_prefix = plot_prefix

        self.X_train = None
        self.y_train = None

    """
    X is the training array and y is the labels. 
    This method does nothing, but some type checking and validation
    and storing the training data and the labels in this class. 
    That's because we need the training data and labels during inference. 
    That's how KNN works. You look into the neighbors and the the most
    frequency of labels within that neighborhood wins. 
    """
    def fit(self, X, y):
        """
        Store the training data and labels as NumPy arrays.

        Requirements:
            - convert X and y to NumPy arrays
            - check that X has shape (n_samples, n_features)
            - check that len(X) == len(y)
            - return self
        """
        """
        X and Y can be anything. A python list or a tuple or something. 
        This np.asarray makes sure that we convert them to numpy arrays 
        before validation and storage. 
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
        We return self because it allows for method chaining. For example:
        predictions = KNNClassifier(n_neighbors=3).fit(X_train, y_train).predict(X_test)
        """
        return self

    def _euclidean_distances(self, x):
        """
        Return the Euclidean distance from x to all training samples.
        Basically this is one of the reasons that they say KNN is not 
        scalable. When the training set is huge, calculating these distances
        is really computationally expensive. 
        """
        return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

    def _cosine_distances(self, x):
        """
        Return the cosine distance from x to all training samples.

        Hint:
            cosine_distance = 1 - cosine_similarity
            cosine_similarity = (a · b) / (||a|| * ||b||)

        Make sure that zero vectors do not cause a division-by-zero error.
        """
        """
        This is not normalizing x, it is calculating its norm. 
        The norm of a vector is its magnitude. AKA, its size which 
        is used in the denominator of the cosine similarity formula. 
        """
        x_norm = np.linalg.norm(x)
        """
        axis=1 means for each row. For each item. Not the norm of the entire vector.
        """
        train_norms = np.linalg.norm(self.X_train, axis=1)
        denominator = train_norms * x_norm

        """
        Pay attention that we used self.X_train here instead of the 
        train_norms because in the formula's numerator the original 
        values are being used for dot product calculation not their magnitude. 
        """
        dot_products = self.X_train @ x

        """
        where=denominator != 0,
        This helps with making sure division by zero does not happen. 
        We are using np.divide instead of standard python division because
        of this exactly. The where and out parameters help with that. 
        """
        similarity = np.divide(
            dot_products,
            denominator,
            out=np.zeros_like(dot_products, dtype=np.float64),
            where=denominator != 0,
        )
        """
        This method only returns the distances not the least distance. 
        That's what the _majority_vote method does. 
        """
        return 1.0 - similarity

    def _majority_vote(self, neighbor_labels):
        """
        Return the most frequent label among the nearest neighbors.
        The predict function is responsible for passing this array
        of nearest neighbors to it. 
        Essentially this method takes a list of the labels of the 
        neighboring values to a point. Then it should return the most
        frequent label. 

        Hint:
            np.unique(..., return_counts=True) is useful here.
            If there is a tie, choose the smallest label after sorting.
        """
        """
        We apply np.unique because we only care about the name of the labels. 
        Basically we are stripping out the counts as a separate array. 
        """
        labels, counts = np.unique(neighbor_labels, return_counts=True)
        """
        labels is an ascending sorted array of the labels and counts is the 
        array that contains the frequency of each of these labels. 
        Have in mind that the labels passed are not strings but numbers. 
        It is important for the labels array to be sorted in ascending order because
        it helps with the tie resolution situations. The tutors have specified that in 
        case of a tie, meaning the same frequency among a set of labels, the one with the 
        least value should be returned. So for example if we have 2 of labels 3 and 7, 3 
        will be returned even though they have the same frequency. 
        """
        """
        np.argmax will return the index of the most recurring label. 
        arg means index. 
        """
        return labels[np.argmax(counts)]

    def predict(self, X):
        """
        Predict labels for one or more input samples.

        Requirements:
            - convert X to a NumPy array. This is the list of of test values. 
            - allow a single sample with shape (n_features,)
            - compute distances with the selected metric
            - select the k nearest neighbors
            - predict by majority vote
            - optionally save neighbor plots when plot_neighbors is True
        """

        X = np.asarray(X, dtype=np.float64)

        if self.X_train is None or self.y_train is None:
            raise ValueError("The classifier must be fitted before calling predict().")

        """
        Do you remember what we did in the last exercise?
        this is exactly the same thing. 
        This if is added to handle cases where only one item is passed as input. (n_features,)
        This code reshapes it to (1, n_features). This way it will be compatible with 
        the rest of the code. Otherwise, KNN will happen on a pixel-based basis. 
        """
        if X.ndim == 1:
            """
            This means you should reshape it to a 1D array and -1 means I don't give a fuck
            how you do it, you figure it out. 
            """
            X = X.reshape(1, -1)

        predictions = []
        """
        enumerate(X) gives us an advantage that not using it and just saying "in X"
        doesn't. 
        """
        for sample_index, x in enumerate(X):
            if self.metric == "euclidean":
                distances = self._euclidean_distances(x)
            elif self.metric == "cosine":
                distances = self._cosine_distances(x)
            else:
                raise ValueError(f"Unsupported metric: {self.metric}")

            """
            np.argsort will give the indices of the sorted array. Not that it sorts it
            but tells each item should go to which index in order to sort the input array. 
            But here we want to extract the number of the K nearest neighbors then in 
            the next line we will get the actual labels of them by passing the indices to the 
            array of the entire training dataset. 
            """
            nn_indices = np.argsort(distances)[: self.n_neighbors]

            nn_labels = self.y_train[nn_indices]

            predicted_label = self._majority_vote(nn_labels)
            predictions.append(predicted_label)
            
            """
            This part is simple. It will save the neighboring images because it is 
            interesting to know exactly what the neighbors where and what was the prediction. 
            These images can help with hyperparameter tuning. 
            """
            if self.plot_neighbors and self.image_shape is not None:
                test_image = x.reshape(self.image_shape)
                neighbor_images = [
                    self.X_train[idx].reshape(self.image_shape) for idx in nn_indices
                ]

                save_path = None
                if self.plot_dir is not None:
                    save_path = (
                        Path(self.plot_dir)
                        / f"{self.plot_prefix}_{sample_index:02d}.png"
                    )
                    
                plot_knn_neighbors(
                    test_image=test_image,
                    neighbor_images=neighbor_images,
                    neighbor_labels=nn_labels,
                    test_label=None,
                    pred_label=predicted_label,
                    save_path=save_path,
                )

        return np.asarray(predictions)
