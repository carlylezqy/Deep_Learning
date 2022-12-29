import mmcv
import numpy as np
import matplotlib.pyplot as plt

def show_images_and_labels(images, labels, size=(28, 28), gray=False):
    _, figs = plt.subplots(1, len(images), figsize=size)
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(np.squeeze(img.numpy()), cmap="gray" if gray else None)
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


def main():
    pass

if __name__ == '__main__':
    main()