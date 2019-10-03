import matplotlib.pyplot as plt

import dataloader


def main():
    datagen = dataloader.get_dataloader('data')
    for x, xo in datagen:
        print(xo.shape)
        axes = plt.gca()
        axes.axis('off')
        axes.imshow(xo[0].numpy())
        plt.show()
        break


if __name__ == '__main__':
    main()
