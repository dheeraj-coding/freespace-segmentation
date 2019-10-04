import matplotlib.pyplot as plt
import torch

import dataloader
from label_generator import SuperPixelAlign
from drn import drn


def main():
    feature_shape = (28, 28, 512)
    img_shape = (224, 224, 3)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    datagen = dataloader.get_dataloader('data', batch_size=1, shuffle=False)
    aligner = SuperPixelAlign(250, 10, 100, 15, feature_shape, img_shape)

    model = drn.drn_c_26(pretrained=True)
    model.cuda()
    model.eval()

    for x, xo in datagen:
        x = x.to(device)
        prediction = model(x)
        prediction = prediction.permute(0, 2, 3, 1)
        prediction = prediction.cpu().detach().numpy()
        superpixel_aligned, segments = aligner.align(xo[0], prediction[0])
        print(superpixel_aligned.shape)
        axes = plt.gca()
        axes.axis('off')
        axes.imshow(segments)
        plt.show()
        break


if __name__ == '__main__':
    main()
