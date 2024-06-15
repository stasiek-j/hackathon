import os
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt


def aggregate_file(file, min_t, max_t, step):
    im = np.zeros_like(file)
    fig, axs = plt.subplots(nrows=1, ncols=len(np.arange(min_t, max_t+step/2, step=step)), figsize=(120, 5))
    ax = np.ravel(axs)
    i = 0
    for thr in np.arange(min_t, max_t+step/2, step=step):
        im += file > thr
        ax[i].imshow(file > thr)
        ax[i].title.set_text('Thr:  {}'.format(thr))
        ax[i].axis('off')
        i += 1
    plt.show()
    return im


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument("input_dir", type=str, help="Input directory.")

    args = args.parse_args()
    input_dir = args.input_dir

    min_t = .1
    max_t = .9
    step = .05

    for filename in os.listdir(input_dir):
        if filename.endswith(".npy"):
            file = os.path.join(input_dir, filename)
            file = np.load(file)
            agg_im = aggregate_file(file, min_t, max_t, step)
            fig, axs = plt.subplots(1, 3, figsize=(10, 3))
            ax = np.ravel(axs)
            c = 'grey'  # 'Spectral'

            ax[0].imshow(file, cmap=c)
            ax[0].title.set_text("Probabilities")
            ax[1].imshow(agg_im, cmap=c)
            ax[1].title.set_text("Aggregated")
            ax[2].imshow(file > .95, cmap=c)
            ax[2].title.set_text(f"Binarized  (trh: {max_t})")

            fig.tight_layout()
            for a in ax:
                a.axis('off')
            plt.show()


