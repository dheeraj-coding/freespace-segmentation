import matplotlib.pyplot as plt
import torch
import numpy as np

import dataloader
from label_generator import SuperPixelAlign, WeightedKMeans
from drn import drn


def main():
    feature_shape = (28, 28, 512)
    img_shape = (224, 224, 3)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    datagen = dataloader.get_dataloader('data', batch_size=1, shuffle=True)
    aligner = SuperPixelAlign(300, 0.5, 50, 10, feature_shape, img_shape)
    kmeans = WeightedKMeans(5, np.array([[0.3, 0.]]), 0.1)

    model = drn.drn_c_26(pretrained=True)
    model.cuda()
    model.eval()

    for x, xo in datagen:
        x = x.to(device)
        prediction = model(x)
        prediction = prediction.permute(0, 2, 3, 1)
        prediction = prediction.cpu().detach().numpy()
        superpixel_aligned, segments = aligner.align(xo[0], prediction[0])
        c, labels = kmeans.fit(superpixel_aligned)
        for j in np.unique(segments):
            segments[segments == j] = labels[j]
        fig, axes = plt.subplots(1, 2)
        axes = axes.flatten()
        axes[0].axis('off')
        axes[1].axis('off')
        axes[0].imshow(segments)
        axes[1].imshow(xo[0])
        plt.show()
        break


if __name__ == '__main__':
    main()
