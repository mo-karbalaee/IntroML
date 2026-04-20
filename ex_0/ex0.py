import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Do not alter this path!
IMAGE_PATH: str = "data/Image01.png"


class ImageProcessor:

    def __init__(self, image_path: str, colour_type: str = "BGR"):
        """
        Load and save the provided image, the image colour type and the image directory.
        Use CV2 to load the image.

        Args:
        image_path (str): Path to the input image.
        colour_type (str): Colour type of the image (BGR, RGB, Gray).
        """
        # Extract the parent directory of the image.
        self._image_directory: str = os.path.dirname(image_path)
        if colour_type not in ["BGR", "RGB", "Gray"]:
            raise ValueError("The given colour is not supported!")

        self._colour_type: str = colour_type
        if self._colour_type == "BGR":
            self._image: np.ndarray = cv2.imread(image_path, cv2.IMREAD_COLOR)
        elif self._colour_type == "RGB":
            image_bgr: np.ndarray = cv2.imread(image_path, cv2.IMREAD_COLOR)
            self._image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        else:
            self._image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    @property
    def image(self):
        return self._image

    def show_image(self):
        """
        Show the loaded image using either matplotlib or CV2.
        """

        if self._colour_type == "Gray":
            plt.imshow(self._image, cmap="gray")
        else:
            plt.imshow(self._image)

        plt.show()
        pass

    def save_image(self, image_title: str):
        """
        Save the loaded image using either matplotlib or CV2.

        Args:
        image_title (str): Title of the image with the corresponding extension.
        """

        # Combine the image parent directory and the given title to create the path for the new image.
        total_image_path: str = os.path.join(self._image_directory, image_title)

        cv2.imwrite(total_image_path, self._image)
        pass

    def convert_colour(self):
        """
        Convert a colour image from BGR to RGB or vice versa.
        Do not use functions from external libraries.
        Solve this task by using indexing.
        """
        if self._colour_type not in ["RGB", "BGR"]:
            raise ValueError("The function only works for colour images!")

        self._image = self._image[:, :, ::-1]

        self._colour_type = "RGB" if self._colour_type == "BGR" else "BGR"

    def clip_image(self, clip_min: int, clip_max: int):
        """
        Clip all colour values in the image to a given min and max value.
        Do not use functions from external libraries.
        Solve this task by using indexing.

        Args:
        clip_min (int): Minimum image colour intensity.
        clip_max (int): Maximum image colour intensity.
        """
        # ToDo: Clip the image values to the given values.
        pass

    def convert_to_grayscale(self, method: str = "lightness"):
        """
        Convert a colour image to a grayscale image.
        Write the different options from scratch.

        Args:
        method (str): Method for the colour conversion, either lightness, average or luminosity.
        """
        if method not in ["lightness", "average", "luminosity"]:
            raise ValueError("The given method is not supported!")
        if self._colour_type not in ["BGR", "RGB"]:
            raise ValueError("The function only works for colour images!")

        if method == "lightness":
            pass

        if method == "average":
            pass

        if method == "luminosity":
            pass

        # ToDo: Update the colour type.
        pass

    def rotate_image(self, degrees: int = 0):
        """
        Rotate an image by a given angle (k * 90) clockwise.
        Do not use functions from external libraries apart from numpy.transpose.

        Args:
        degrees (int): Rotation angle.
        """
        if degrees % 90 != 0:
            raise ValueError("The provided rotation angle must be a multiple of 90!")

        # ToDo: Rotate the image depending on the given rotation value.
        pass

    def flip_image(self, flip_value: int):
        """
        Flip an image either vertically (0), horizontally (1) or both ways (2).
        Do not use functions from external libraries.

        Args:
        flip_value (int): Value to determine how the image should be flipped.
        """
        if flip_value not in [0, 1, 2]:
            raise ValueError("The provided flip value must be either 0, 1 or 2!")

        # ToDo: Flip the image using indexing.
        pass

    def crop_center(self, new_height: int, new_width: int):
        """
        Crop the image to a given size around the center.
        Do not use functions from external libraries.

        Args:
        new_height (int): Height of the cropped image.
        new_width (int): Width of the cropped image.
        """
        # ToDo: Check that the given parameters are valid!
        pass

        # ToDo: Crop the image around the center.
        pass

    def resize_image(self, new_height: int, new_width: int):
        """
        Resize an image to an arbitrary size using CV2.

        Args:
        new_height (int): Height of the resized image.
        new_width (int): Width of the resized image.
        """
        # ToDo: Resize the image. Research the available options in CV2.


if __name__ == '__main__':
    processor = ImageProcessor(image_path=IMAGE_PATH, colour_type="BGR")
