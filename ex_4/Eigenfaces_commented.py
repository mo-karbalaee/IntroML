from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

N = 64  # every image gets resized to N x N pixels (64x64) before anything else happens
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}  # file types we're allowed to load as images
TRAINED_CLASSIFIERS = {}  # a "cache" dict: after training, stores the fitted classifier objects, keyed by name like "logistic" or "gaussian_nb"
TRAINED_STANDARDIZATION = {}  # a "cache" dict: stores (mean, std) computed on the training set, so test images can reuse the exact same numbers later
# Do not add further imports.
# Implement PCA and feature standardization with NumPy only.
# Do not use sklearn.decomposition.PCA or other pre-built standardization helpers.


def _build_classifier(classifier_type):
    # factory function: given a string name, return a fresh, untrained sklearn classifier object
    if classifier_type == "logistic":
        return LogisticRegression(max_iter=2000)  # max_iter raised because logistic regression's solver can need many steps to converge
    if classifier_type == "gaussian_nb":
        return GaussianNB()
    raise ValueError(f"Unknown classifier type: {classifier_type}")  # guard against typos like "logisitc"


def _uses_feature_scaling(classifier_type):
    # only logistic regression needs standardized (zero mean/unit variance) input features
    # GaussianNB estimates its own per-class mean/variance internally, so it doesn't need this
    return classifier_type == "logistic"


def _list_class_directories(dataset_root):
    # a "class directory" is a subfolder whose name IS the label, e.g. dataset_root/cat/, dataset_root/dog/
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
    labels = []  # will hold one string label per image, e.g. "class_a", "class_a", "class_b", ...
    train = []   # will hold one flattened pixel vector per image
    class_dirs = _list_class_directories(dataset_root)  # e.g. [Path("class_a"), Path("class_b")]

    target_height = None
    target_width = None
    if image_size is not None:
        target_width, target_height = image_size  # unpack (width, height) tuple, e.g. (64, 64)

    for class_dir in class_dirs:  # loop over each class folder
        image_paths = [
            path
            for path in sorted(class_dir.iterdir())
            if path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]  # collect only image files inside this folder, skip anything else
        if not image_paths:
            raise ValueError(
                f"Class directory contains no supported images: {class_dir}"
            )

        for image_path in image_paths:  # loop over every image file inside this class folder
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)  # load as a 2D grayscale array (values 0-255)
            if img is None:
                raise FileNotFoundError(f"Could not load image: {image_path}")

            if image_size is not None:
                img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)  # force every image to the same fixed size
            elif target_height is None or target_width is None:
                target_height, target_width = img.shape  # first image seen defines the expected size for all others
            elif (img.shape[1], img.shape[0]) != (target_width, target_height):
                raise ValueError(
                    "All images must share the same size when image_size is None. "
                    f"Expected {(target_width, target_height)}, got {(img.shape[1], img.shape[0])} from {image_path}."
                )

            train.append(img.reshape(-1).astype(np.float64))  # flatten the 2D image (h, w) into a 1D vector (h*w,)
            labels.append(class_dir.name)  # the folder name becomes this image's label

    if not train:
        raise ValueError(f"No images found below {dataset_root}")

    train = np.asarray(train, dtype=np.float64)  # stack all image vectors into one big matrix, shape (n_images, n_pixels)
    return np.asarray(labels), train, train.shape[0], target_height, target_width
    # returns: labels array, the data matrix, number of images, image height, image width


def calculate_average_face(train):
    """
    Calculate the average image using all training images.
    """
    # train has shape (n_images, n_pixels)
    # axis=0 averages DOWN the rows, i.e. one average PER PIXEL POSITION, across all images
    # result: a single vector of length n_pixels, i.e. the "average image"
    return np.mean(train, axis=0)


def calculate_eigenfaces(train, avg, num_eigenfaces):
    """
    Calculate the principal directions of the centered training set using SVD.
    """
    centered = train - avg  # subtract the average face from every image (broadcasting avg across all rows)
    # SVD decomposes centered = U @ diag(S) @ Vt
    # full_matrices=False keeps the shapes small: U is (n_images, k), S is (k,), Vt is (k, n_pixels)
    # where k = min(n_images, n_pixels)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    # the rows of Vt are the principal directions ("eigenfaces"), sorted from most to least variance explained
    # we keep only the first num_eigenfaces of them
    return Vt[:num_eigenfaces]


def get_feature_representation(images, eigenfaces, avg, num_eigenfaces):
    """
    Project all images into the PCA space spanned by the first num_eigenfaces components.
    """
    centered = images - avg  # center the images the same way as during eigenface computation
    # matrix-multiply centered images by the transposed eigenfaces
    # this projects every image onto each of the num_eigenfaces directions,
    # producing one coefficient (a number) per direction per image
    return centered @ eigenfaces[:num_eigenfaces].T


def calculate_feature_statistics(features):
    """
    Compute the mean and standard deviation of every PCA feature over the training set.

    Standardize every feature by centering it to zero mean
    and rescaling it to unit standard deviation. Implement this with NumPy only.
    """
    mean = np.mean(features, axis=0)  # one mean value per PCA feature/column, averaged over all training images
    std = np.std(features, axis=0)    # one standard deviation per PCA feature/column
    std = np.where(std == 0, 1.0, std)  # avoid dividing by zero later: if a feature never varies, treat its std as 1
    return mean, std


def standardize_features(features, feature_mean, feature_std):
    """
    Standardize all features using the previously computed mean and standard deviation.

    Apply the same transformation to the training features and later to every test image.
    """
    # standard z-score formula, applied per feature column: (x - mean) / std
    # feature_mean and feature_std MUST come from the training set, even when standardizing test data
    return (features - feature_mean) / feature_std


def reconstruct_image(img, eigenfaces, avg, num_eigenfaces, h, w):
    """
    Reconstruct an image from the first num_eigenfaces principal components.
    """
    centered = img - avg  # center the input image
    coeffs = centered @ eigenfaces[:num_eigenfaces].T  # project it onto the top num_eigenfaces directions (same as get_feature_representation)
    reconstructed = avg + coeffs @ eigenfaces[:num_eigenfaces]  # map the coefficients back into pixel space and add the average back
    return reconstructed.reshape(h, w)  # reshape the flat pixel vector back into a 2D image for viewing


def process_and_train(
    labels, train, num_images, h, w, classifier_type="logistic", num_eigenfaces=None
):
    """
    Compute PCA features and train one classifier on top of them.
    For Logistic Regression, standardize the PCA features with your own helper functions.
    """
    if num_eigenfaces is None:
        num_eigenfaces = min(num_images, h * w)  # can't have more components than samples or pixels allow

    avg = calculate_average_face(train)  # step 1: compute the average image
    eigenfaces = calculate_eigenfaces(train, avg, num_eigenfaces)  # step 2: compute the principal directions
    features = get_feature_representation(train, eigenfaces, avg, num_eigenfaces)  # step 3: turn every image into its PCA coefficients

    classifier = _build_classifier(classifier_type)  # step 4: create an untrained classifier of the requested type

    if _uses_feature_scaling(classifier_type):
        # logistic regression: standardize the features first
        feature_mean, feature_std = calculate_feature_statistics(features)
        features_input = standardize_features(features, feature_mean, feature_std)
        TRAINED_STANDARDIZATION[classifier_type] = (feature_mean, feature_std)  # save these so classify_image can reuse them later
    else:
        # gaussian_nb: use the raw PCA coefficients, no standardization
        features_input = features

    classifier.fit(features_input, labels)  # step 5: train the classifier
    TRAINED_CLASSIFIERS[classifier_type] = classifier  # save the trained classifier for later use in classify_image

    return eigenfaces, num_eigenfaces, avg  # caller needs these to project/reconstruct/classify new images later


def train_both_classifiers(labels, train, num_images, h, w, num_eigenfaces=None):
    """
    Train Logistic Regression and Gaussian Naive Bayes on the same PCA features.
    For Logistic Regression, standardize the PCA features with your own helper functions.
    """
    if num_eigenfaces is None:
        num_eigenfaces = min(num_images, h * w)

    # PCA is computed only once and shared between both classifiers, so they're compared on identical features
    avg = calculate_average_face(train)
    eigenfaces = calculate_eigenfaces(train, avg, num_eigenfaces)
    features = get_feature_representation(train, eigenfaces, avg, num_eigenfaces)

    # logistic regression branch: needs standardized features
    feature_mean, feature_std = calculate_feature_statistics(features)
    features_standardized = standardize_features(features, feature_mean, feature_std)
    TRAINED_STANDARDIZATION["logistic"] = (feature_mean, feature_std)

    lr = _build_classifier("logistic")
    lr.fit(features_standardized, labels)
    TRAINED_CLASSIFIERS["logistic"] = lr

    # gaussian naive bayes branch: uses the same PCA features, but unstandardized
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
    )  # reshape(1, -1) turns a single flat image vector into a matrix with 1 row, since get_feature_representation expects a matrix

    if _uses_feature_scaling(classifier_type):
        if classifier_type in TRAINED_STANDARDIZATION:
            # normal case: reuse the mean/std that were computed on the TRAINING set
            feature_mean, feature_std = TRAINED_STANDARDIZATION[classifier_type]
        else:
            # fallback safety net: nothing was trained yet, so compute stats from this single image
            # (not very meaningful with just one sample, but prevents a crash)
            feature_mean, feature_std = calculate_feature_statistics(features)
        features = standardize_features(features, feature_mean, feature_std)

    if classifier_type not in TRAINED_CLASSIFIERS:
        return np.array(["unknown"])  # no trained classifier of this type exists yet

    return TRAINED_CLASSIFIERS[classifier_type].predict(features)  # ask the trained classifier to predict the label
