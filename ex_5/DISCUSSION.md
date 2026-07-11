# Exercise 5 – Short Discussion

Accuracies on the full official test split (`images/test`, 968 images, 11 classes):

| Classifier            | Accuracy |
|-----------------------|----------|
| KNN (euclidean)       | 0.7335   |
| KNN (cosine)          | 0.7097   |
| NBNN (euclidean)      | 0.7696   |
| NBNN (cosine)         | 0.7552   |
| Logistic Regression   | 0.9070   |

## Observations

**Euclidean vs. cosine.** For both KNN and NBNN the two distance measures behaved
very similarly: the accuracies differ by only ~2–3 percentage points, with the
Euclidean variant slightly ahead in each case. Because the Kaktovik images are
grayscale symbols normalized to `[0, 1]` and share a similar overall
brightness/scale, the magnitude information that cosine distance discards is not
very discriminative here, so neither metric gains a clear advantage.

**KNN vs. NBNN.** NBNN edged out KNN under both metrics (e.g. 0.770 vs. 0.734 for
Euclidean). NBNN scores each class by its single nearest training sample rather
than by a majority vote over the `k=3` global neighbours, so it is less affected
by classes that happen to be densely represented in the shared neighbourhood.

**Best classifier.** Logistic Regression was the clear winner at ~0.91, well above
all nearest-neighbour variants. Learning a linear decision boundary in pixel space
generalizes noticeably better on this dataset than the purely instance-based
methods, which rely directly on raw pixel distances and are more sensitive to the
rotation augmentations present in the data.
