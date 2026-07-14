import numpy as np


class NBNNClassifier:
    def __init__(self, metric="euclidean", patch_size=16, stride=16, image_shape=None):
        self.metric = metric
        self.patch_size = patch_size
        self.stride = stride
        self.image_shape = image_shape

        self.X_train = None
        self.y_train = None
        self.classes_ = None

        self._use_patches = False
        self._image_shape = None
        self._patch_descriptors = None
        self._patch_labels = None
        self._class_columns = None
        self._patch_sq_norms = None

    def _resolve_image_shape(self, n_features):
        if self.image_shape is not None:
            return self.image_shape
        side = int(round(np.sqrt(n_features)))
        if side * side == n_features:
            return (side, side)
        return None

    def _extract_patches(self, row):
        h, w = self._image_shape
        image = row.reshape(h, w)
        p = self.patch_size
        s = self.stride
        patches = []
        for i in range(0, h - p + 1, s):
            for j in range(0, w - p + 1, s):
                patches.append(image[i : i + p, j : j + p].reshape(-1))
        return np.asarray(patches, dtype=np.float64)

    def _normalize_rows(self, matrix):
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)

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

        shape = self._resolve_image_shape(X.shape[1])
        self._image_shape = shape
        self._use_patches = (
            shape is not None
            and self.patch_size <= shape[0]
            and self.patch_size <= shape[1]
        )

        if self._use_patches:
            descriptor_blocks = []
            label_blocks = []
            for row, label in zip(X, y):
                patches = self._extract_patches(row)
                descriptor_blocks.append(patches)
                label_blocks.append(np.full(len(patches), label))
            descriptors = np.concatenate(descriptor_blocks, axis=0)
            labels = np.concatenate(label_blocks, axis=0)
        else:
            descriptors = X
            labels = y

        if self.metric == "cosine":
            descriptors = self._normalize_rows(descriptors)

        self._patch_descriptors = descriptors
        self._patch_labels = labels
        self._patch_sq_norms = np.sum(descriptors**2, axis=1)
        self._class_columns = [np.where(labels == cls)[0] for cls in self.classes_]
        return self

    def _descriptor_distances(self, query):
        train = self._patch_descriptors
        if self.metric == "cosine":
            query = self._normalize_rows(query)
            return 1.0 - query @ train.T
        elif self.metric == "euclidean":
            query_sq = np.sum(query**2, axis=1, keepdims=True)
            d2 = query_sq + self._patch_sq_norms[None, :] - 2.0 * (query @ train.T)
            np.maximum(d2, 0.0, out=d2)
            return np.sqrt(d2)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def _class_scores(self, distances):
        scores = np.empty(len(self.classes_), dtype=np.float64)
        for index, columns in enumerate(self._class_columns):
            scores[index] = distances[:, columns].min(axis=1).sum()
        return scores

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)

        if self.X_train is None or self.y_train is None:
            raise ValueError("The classifier must be fitted before calling predict().")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictions = []
        for row in X:
            if self._use_patches:
                query = self._extract_patches(row)
            else:
                query = row.reshape(1, -1)
            distances = self._descriptor_distances(query)
            scores = self._class_scores(distances)
            predictions.append(self.classes_[np.argmin(scores)])

        return np.asarray(predictions)
