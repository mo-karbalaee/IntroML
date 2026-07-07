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
    
    The output is of shape (d, ) where d is h * w. It makes sense that the average image is 
    of the same shape as the rest of the images in the dataset right? 
    axis=0 means, go down each column for all rows and calculate the average of that. 
    For example: 
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ]
    
    The output = [(1 + 2 + 3) / 3, (2 + 6 + 10) / 3, ....., ]
    """
    return np.mean(train, axis=0)


def calculate_eigenfaces(train, avg, num_eigenfaces):
    """
    Calculate the principal directions of the centered training set using SVD (singular value decomposition).
    Another method of performing PCA is using covariance matrices, but that is not computationally 
    viable because calculating covariance matrices for high-resolution pictures is not efficient. 
    Instead, we can do it with SVD. 
    """
    centered = train - avg
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    """
    We only need the first num_eigenfaces of principal components. Vt stores
    these eigenvectors sorted by importance (how big their eigenvalues are). Since
    we don't want all the principal components, we just slice this array and return 
    the sub-array we are interested in. 
    """
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
    This is just a z-score normalization. 
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
    This functions is doing nothing new. Just trains only one classifier instead of both of them. 
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
    Training set:
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ]
    
    labels: [Yale1, Yale10, Yale4]
    """

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

    """
    Since we are only concerned with the variations in our dataset, we need to 
    calculate the average of our dataset and subtract it from the training set. That's
    done in PCA. The reason we do this is that it makes our training set zero-centered which
    actually allows the PCA to lean the variations. How? because if we don't subtract the mean,
    Then the first principal component will point to the direction of the mean training data. 
    Which is not important for us. We want PC-reduced images to contain key information about
    themselves not the entire dataset. That actually makes it more difficult for our classifier
    to differentiate between different images. Makes sense right?
    """
    avg = calculate_average_face(train)
    """
    Here we calculate the principal components in the training set. Meaning, we detect different features
    in the training set. We haven't described the images in our training set using those features yet. 
    Think of it this way, in this step we only calculate features like Asian eyes or beard or long hair 
    and things like this across the entire training set. That's it. For example, we say that the top two
    prominent features of the images in this dataset is having a beard and having Asian eyes. 
    """
    eigenfaces = calculate_eigenfaces(train, avg, num_eigenfaces)
    """
    Now that we know the prominent features in our dataset, it's time to see how much of those features
    each of the images in our training set posses. This is the part that condenses our images from 
    h * w dimensions down to num_eigenfaces dimensions. How? by projection. 
    So each principal component is a vector and here each of our images are a vector too. 
    By projecting each image to each of those principal components gives us one scalar per principal 
    component. So if we have 50 principal components, we'll have 50 scalars. Each of them tells us
    how much our image aligns with such feature. How much Asian eyes is in it. Or how much beard is in it. 
    """
    features = get_feature_representation(train, eigenfaces, avg, num_eigenfaces)

    """
    Just calculating the mean and standard deviation of the features (our images described in terms of
    principal components.) later on used for standardizing all those features. Nothing fancy. 
    """
    feature_mean, feature_std = calculate_feature_statistics(features)
    """
    The following line standardizes the features. What does it mean? It means that it ensures
    the mean of the data will be 0 and the standard deviation will be 1. This type of standardization 
    has another name. They also call it z-score normalization. 
    """
    features_standardized = standardize_features(features, feature_mean, feature_std)
    """
    We save the mean and std her because upon inference, we should use the same values that we used 
    during training. The reason we care is that if we recalculate the std and mean of the test set 
    and we standardize using these values, it is a form of data leakage. Which means, we are cheating 
    to get better results. The problem is that during inference in real deployments we don't have a 
    predefined huge test set that we can standardize our input with then perform inference on each of them,
    but rather, each new image comes right away. So we don't want to report an accuracy that relies on having
    a huge test set. That is misleading. 
    """
    TRAINED_STANDARDIZATION["logistic"] = (feature_mean, feature_std)

    lr = _build_classifier("logistic")
    """
    Because logistic regression learns the weights and adjusts them iteratively right? this works better
    in standardized data. The training will go smooth. 
    """
    lr.fit(features_standardized, labels)
    """
    We save the model so that we can use it during inference. 
    """
    TRAINED_CLASSIFIERS["logistic"] = lr

    gnb = _build_classifier("gaussian_nb")
    """
    Gaussian Naive Bayes classifier does not standardized features though. 
    GNB fits a separate bell curve (mean and variance) to each feature within each class, 
    and judges each feature only against its own curve — it never combines features using shared 
    weights the way logistic regression does. Since there's no cross-feature weighting to balance, 
    the scale of one feature relative to another simply doesn't matter for how well the model fits or predicts.
    """
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
        """
        Why do we need to standardize before inference?
        """    
        features = standardize_features(features, feature_mean, feature_std)

    if classifier_type not in TRAINED_CLASSIFIERS:
        return np.array(["unknown"])

    return TRAINED_CLASSIFIERS[classifier_type].predict(features)
