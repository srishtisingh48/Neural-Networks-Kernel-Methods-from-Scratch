if __name__ == "__main__":
    from k_means import lloyd_algorithm  # type: ignore
else:
    from .k_means import lloyd_algorithm

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


def main():
    """Main function of k-means problem

    Run Lloyd's Algorithm for k=10, and report 10 centers returned.

    NOTE: This code might take a while to run. For debugging purposes you might want to change:
        x_train to x_train[:10000]. Make sure to change it back before submission!
    """
    (x_train, _), _ = load_dataset("mnist")

    x_train = x_train.reshape((-1, 28 * 28)) / 255.0
    centers, errors = lloyd_algorithm(x_train, num_centers=10)
    for i, center in enumerate(centers):
        plt.subplot(1, 10, i + 1)
        plt.imshow(center.reshape(28, 28) , cmap="gray")
        plt.axis("off") 
    plt.suptitle("K-means Cluster Centers")
    plt.savefig("5.png")
    plt.show()

if __name__ == "__main__":
    main()
