from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

N = 64
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
TRAINED_CLASSIFIERS = {}
TRAINED_STANDARDIZATION = {}
# Do not add further imports.
# Implement PCA and feature standardization with NumPy only.
# Do not use sklearn.decomposition.PCA or other pre-built standardization helpers.


def _build_classifier(classifier_type):
    if classifier_type == "logistic":
        return LogisticRegression(max_iter=2000)
    if classifier_type == "gaussian_nb":
        return GaussianNB()
    raise ValueError(f"Unknown classifier type: {classifier_type}")


def _uses_feature_scaling(classifier_type):
    return classifier_type == "logistic"


def _list_class_directories(dataset_root):
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    class_dirs = [path for path in sorted(dataset_root.iterdir()) if path.is_dir()]
    if not class_dirs:
        raise ValueError(
            f"Expected at least one class subdirectory in {dataset_root}. "
            "Use a structure like dataset/class_name/image.png."
        )
    return class_dirs


def create_database_from_folder(dataset_root, image_size=(N, N)):
    """
    Load a local image dataset from class subdirectories.

    Expected structure:
        dataset_root/
            class_a/
                img_01.png
            class_b/
                img_02.png
    """
    labels = []
    train = []
    class_dirs = _list_class_directories(dataset_root)

    target_height = None
    target_width = None
    if image_size is not None:
        target_width, target_height = image_size

    for class_dir in class_dirs:
        image_paths = [
            path
            for path in sorted(class_dir.iterdir())
            if path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not image_paths:
            raise ValueError(
                f"Class directory contains no supported images: {class_dir}"
            )

        for image_path in image_paths:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Could not load image: {image_path}")

            if image_size is not None:
                img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
            elif target_height is None or target_width is None:
                target_height, target_width = img.shape
            elif (img.shape[1], img.shape[0]) != (target_width, target_height):
                raise ValueError(
                    "All images must share the same size when image_size is None. "
                    f"Expected {(target_width, target_height)}, got {(img.shape[1], img.shape[0])} from {image_path}."
                )

            train.append(img.reshape(-1).astype(np.float64))
            labels.append(class_dir.name)

    if not train:
        raise ValueError(f"No images found below {dataset_root}")

    train = np.asarray(train, dtype=np.float64)
    return np.asarray(labels), train, train.shape[0], target_height, target_width


def calculate_average_face(train):
    """
    Calculate the average image using all training images.
    We need this method because PCA works on centered data which means that
    the average faces will be deducted from the training batch.
    The output of this function is a 1D vector of average pixels. Which means that
    the following line will calculate the average pixel intensity for the pixels on the
    same position across the training set. 
    """
    return np.mean(train, axis=0)


def calculate_eigenfaces(train, avg, num_eigenfaces):
    """
    Calculate the principal directions of the centered training set using SVD.
    """
    centered = train - avg
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    return Vt[:num_eigenfaces]


def get_feature_representation(images, eigenfaces, avg, num_eigenfaces):
    """
    Project all images into the PCA space spanned by the first num_eigenfaces components.
    """
    centered = images - avg
    return centered @ eigenfaces[:num_eigenfaces].T


def calculate_feature_statistics(features):
    """
    Compute the mean and standard deviation of every PCA feature over the training set.

    Standardize every feature by centering it to zero mean
    and rescaling it to unit standard deviation. Implement this with NumPy only.
    """
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std = np.where(std == 0, 1.0, std)
    return mean, std


def standardize_features(features, feature_mean, feature_std):
    """
    Standardize all features using the previously computed mean and standard deviation.

    Apply the same transformation to the training features and later to every test image.
    """
    return (features - feature_mean) / feature_std


def reconstruct_image(img, eigenfaces, avg, num_eigenfaces, h, w):
    """
    Reconstruct an image from the first num_eigenfaces principal components.
    """
    centered = img - avg
    coeffs = centered @ eigenfaces[:num_eigenfaces].T
    reconstructed = avg + coeffs @ eigenfaces[:num_eigenfaces]
    return reconstructed.reshape(h, w)


def process_and_train(
    labels, train, num_images, h, w, classifier_type="logistic", num_eigenfaces=None
):
    """
    Compute PCA features and train one classifier on top of them.
    For Logistic Regression, standardize the PCA features with your own helper functions.
    """
    if num_eigenfaces is None:
        num_eigenfaces = min(num_images, h * w)

    avg = calculate_average_face(train)
    eigenfaces = calculate_eigenfaces(train, avg, num_eigenfaces)
    features = get_feature_representation(train, eigenfaces, avg, num_eigenfaces)

    classifier = _build_classifier(classifier_type)

    if _uses_feature_scaling(classifier_type):
        feature_mean, feature_std = calculate_feature_statistics(features)
        features_input = standardize_features(features, feature_mean, feature_std)
        TRAINED_STANDARDIZATION[classifier_type] = (feature_mean, feature_std)
    else:
        features_input = features

    classifier.fit(features_input, labels)
    TRAINED_CLASSIFIERS[classifier_type] = classifier

    return eigenfaces, num_eigenfaces, avg


def train_both_classifiers(labels, train, num_images, h, w, num_eigenfaces=None):
    """
    Train Logistic Regression and Gaussian Naive Bayes on the same PCA features.
    For Logistic Regression, standardize the PCA features with your own helper functions.
    # Inputs
    labels: A numpy array containing the labels for each image in the training set.
    train: The training set. It's a matrix where each row of it is a flattened image of size h * W.
            The labels and and train are matched element-wise. So the first item in the labels is the
            label for the first row (image) in train.
    num_images: Number of images in the training set. This parameter is redundant because it could be
                easily calculated inside the function using train.shape[0]
    h: height of the images in the training set. 
    w: width of the images in the training set. 
    num_eigenfaces: The number of principal components to keep per image.             
    """

    if num_eigenfaces is None:
        """
        PCA cannot produce more principal components than the number of independent
        directions in the data (i.e., the rank of the centered data matrix).
        The tank of a matrix is the number of independent columns or row in it.
        A matrix is said to be full rank when all of its rows or columns are
        independent. This means, that a column cannot be created by a linear
        combination of other columns. This means, all of the columns in this data
        have new information. So the rank of a matrix is the number of its important
        columns. The rest is trash. You can remove it and live the rest. And generate
        the whole thing using your independent columns. You just need to combine them.
        Our train data is of shape (num_images, h * w) and the rank of this matrix.
        Cannot be bigger than min(num_images, h * w). Clearly. right?
        So rank(train) = min(num_images, h * w) at most. That is the higher bound. 
        Under the assumption that the train matrix is full-rank. 
        """

        num_eigenfaces = min(num_images, h * w)

    avg = calculate_average_face(train)
    eigenfaces = calculate_eigenfaces(train, avg, num_eigenfaces)
    features = get_feature_representation(train, eigenfaces, avg, num_eigenfaces)

    feature_mean, feature_std = calculate_feature_statistics(features)
    features_standardized = standardize_features(features, feature_mean, feature_std)
    TRAINED_STANDARDIZATION["logistic"] = (feature_mean, feature_std)

    lr = _build_classifier("logistic")
    lr.fit(features_standardized, labels)
    TRAINED_CLASSIFIERS["logistic"] = lr

    gnb = _build_classifier("gaussian_nb")
    gnb.fit(features, labels)
    TRAINED_CLASSIFIERS["gaussian_nb"] = gnb

    return eigenfaces, num_eigenfaces, avg


def classify_image(
    img, eigenfaces, avg, num_eigenfaces, h, w, classifier_type="logistic"
):
    """
    Predict the class label of one image from its PCA coefficients.
    If Logistic Regression is used, apply the same feature standardization as during training.
    """
    features = get_feature_representation(
        img.reshape(1, -1), eigenfaces, avg, num_eigenfaces
    )

    if _uses_feature_scaling(classifier_type):
        if classifier_type in TRAINED_STANDARDIZATION:
            feature_mean, feature_std = TRAINED_STANDARDIZATION[classifier_type]
        else:
            feature_mean, feature_std = calculate_feature_statistics(features)
        features = standardize_features(features, feature_mean, feature_std)

    if classifier_type not in TRAINED_CLASSIFIERS:
        return np.array(["unknown"])

    return TRAINED_CLASSIFIERS[classifier_type].predict(features)
