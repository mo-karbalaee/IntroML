import matplotlib.pyplot as plt
import numpy as np

def show_image(image:np.ndarray, title:str):
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def show_histogram(histogram:np.ndarray):
    bin_edges = np.arange(256)

    plt.figure(figsize=(8, 4))
    plt.title("Grayscale Histogram")
    plt.xlabel("Pixel Intensity (0-255)")
    plt.ylabel("Number of Pixels")

    plt.bar(bin_edges, histogram, color='blue', width=1.0)

    plt.xlim([0, 255])

    plt.show()