import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


URL = './61554659_605.jpg'


def method1(
    img: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    width = img.shape[0]
    height = img.shape[1]

    img = img.reshape(width * height, 3)

    kmeans = KMeans(n_clusters=n_clusters).fit(img)
    labels = kmeans.predict(img)
    clusters = kmeans.cluster_centers_

    img2 = np.zeros_like(img)

    for i, lable in enumerate(labels):
        img2[i] = clusters[lable]

    return img2.reshape(width, height, 3)


def compress_imgage(
    img: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    width = img.shape[0]
    height = img.shape[1]

    img = img.reshape(width * height, 3)

    kmeans = KMeans(n_clusters=n_clusters).fit(img)
    labels = kmeans.predict(img)
    clusters = kmeans.cluster_centers_

    img2 = np.zeros((width, height, 3), dtype=np.uint8)

    i_label = 0
    for i in range(width):
        for j in range(height):
            label_of_pixel = labels[i_label]
            img2[i][j] = clusters[label_of_pixel]
            i_label += 1

    return img2


def main():
    img = plt.imread(URL)
    comp_img = method1(img, n_clusters=4)
    plt.imshow(comp_img)
    plt.show()


if __name__ == '__main__':
    main()
