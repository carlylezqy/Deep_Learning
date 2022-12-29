import mmcv
import numpy as np
import matplotlib.pyplot as plt

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def show_images_and_labels(images, labels, size=(28, 28), gray=False, output_path=None):
    _, figs = plt.subplots(1, len(images), figsize=size)
    for f, img, lbl in zip(figs, images, labels):
        image = img.numpy().transpose((1, 2, 0))
        image = np.uint8(normalization(image) * 255)
        f.imshow(np.squeeze(image), cmap="gray" if gray else None)
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    
    plt.savefig(output_path)
    plt.close()

def main():
    pass

if __name__ == '__main__':
    main()