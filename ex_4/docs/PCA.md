# Understanding PCA (with Images as a Running Example)

PCA is one of those algorithms that becomes much easier once you understand *why* each step exists. We'll go through it slowly, using images as our running example.

---

## Goal of PCA

Suppose you have:

- $n$ images
- each image is $h \times w$

Each image has

$$d = h \times w$$

pixels.

Instead of describing each image with all $d$ pixels, we'd like to describe it using only a few numbers while preserving as much information (variance) as possible.

For example:

- Original image: 10,000 pixels
- PCA representation: 50 numbers

That's dimensionality reduction.

---

## Step 1. Arrange the Data into a Matrix

Flatten every image into a vector.

If an image is

$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$$

it becomes

$$[1, 2, 3, 4]$$

If you have $n$ images, stack them:

$$
X =
\begin{bmatrix}
- & x_1^T & - \\
- & x_2^T & - \\
 & \vdots & \\
- & x_n^T & -
\end{bmatrix}
$$

where $X \in \mathbb{R}^{n \times d}$.

- Rows = images
- Columns = pixels

**Why?** PCA treats every image as a point in a $d$-dimensional space. For a $100 \times 100$ image, that's a point in $\mathbb{R}^{10000}$.

---

## Step 2. Compute the Mean Image

For every pixel position, compute the average value across all images.

For example, suppose three images have first-pixel values:

- 20
- 30
- 40

The average is 30.

Do this for every pixel. You obtain the mean image $\mu$. It has the same size as one image.

**Why?** The mean image tells us the "typical" image in the dataset. But PCA is not interested in the average — it is interested in: *how does each image differ from the average?*

---

## Step 3. Center the Data

Subtract the mean image from every image:

$$X_c = X - \mu$$

Each row becomes $x_i - \mu$.

**Why?** Imagine everyone's height is increased by 100 cm. The variation among people hasn't changed — only the reference point has shifted. PCA studies variation, so we remove the common offset by centering the data. After centering, the average of each feature (pixel) is zero.

---

## Step 4. Compute the Covariance Matrix

Now compute:

$$C = \frac{1}{n-1} X_c^T X_c$$

The covariance matrix has size $d \times d$. Each entry tells you: *do these two pixels tend to brighten or darken together?*

**Why?** The covariance matrix summarizes the variability of the entire dataset. PCA will use this to identify the directions along which the data changes the most.

---

## Step 5. Compute Eigenvectors and Eigenvalues

Now solve:

$$C v = \lambda v$$

For every solution:

- $v$ is an eigenvector
- $\lambda$ is the corresponding eigenvalue

### What do the eigenvectors mean?

Each eigenvector is a direction in the $d$-dimensional feature space. For images, it's an image-sized pattern that tells us a particular way pixel intensities tend to vary together.

For face images, an eigenvector might represent a pattern where:

- the forehead and cheeks brighten together, while
- the eye region darkens.

These are often called **eigenfaces** in face recognition.

### What do the eigenvalues mean?

Each eigenvalue tells you how much of the dataset's variance lies along its corresponding eigenvector. A larger eigenvalue means the data varies more in that direction.

---

## Step 6. Sort by Eigenvalue

You may obtain many eigenvectors. Sort them from largest eigenvalue to smallest:

$$\lambda_1 \geq \lambda_2 \geq \lambda_3 \geq \dots$$

The first eigenvector explains the most variance. The second explains the next most. And so on.

**Why?** We want to keep the directions that carry the most information and discard those that contribute little variation.

---

## Step 7. Keep Only the First $k$ Principal Components

Instead of using all $d$ eigenvectors, choose only the top $k$:

$$W = [v_1, v_2, \dots, v_k]$$

For example:

- original dimension = 10,000
- keep only 100 principal components

Now $W \in \mathbb{R}^{d \times 100}$.

**Why?** These $k$ directions capture most of the variation in the data, allowing us to represent images much more compactly.

---

## Step 8. Project the Data

Compute:

$$Z = X_c W$$

Now $Z \in \mathbb{R}^{n \times k}$. Each image is represented by just $k$ numbers instead of $d$ pixels. These numbers are called the **principal component scores**.

### What does projection mean?

Each principal component is a direction (an eigenvector). Projecting an image onto that direction answers: *"How much of this pattern is present in the image?"*

If an eigenvector resembles a "smiling" facial pattern, an image with a large projection onto it exhibits that pattern strongly.

---

## Step 9. (Optional) Reconstruct the Image

You can approximately reconstruct an image using:

$$\hat{X} = ZW^T + \mu$$

If $k$ is much smaller than $d$, the reconstruction won't be perfect, but it often preserves the main structure while discarding noise.

---

## The Big Picture

You can think of PCA as the following pipeline:

```
Images
   │
   ▼
Flatten each image
   │
   ▼
Data matrix X
   │
   ▼
Subtract the mean
   │
   ▼
Centered data Xc
   │
   ▼
Covariance matrix
   │
   ▼
Eigenvectors + Eigenvalues
   │
   ▼
Sort by eigenvalue
   │
   ▼
Keep top k eigenvectors
   │
   ▼
Project data
   │
   ▼
Compressed representation (k numbers per image)
```

## Intuition in One Sentence

PCA asks:

> "What are the few orthogonal directions in which my data varies the most?"

It then expresses every sample as a weighted combination of those directions. This is why PCA is effective for compression, visualization, denoising, and as a preprocessing step for many machine learning algorithms.

A natural next step is to work through a small numerical example (e.g., 5 points in 2D). Seeing PCA operate on a simple dataset makes concepts like covariance, eigenvectors, and projection much more concrete before applying them to high-dimensional images.
