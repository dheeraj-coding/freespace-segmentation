import dataloader
from label_generator import LabelGenerator
from drn import drn


def main():
    feature_shape = (28, 28, 512)
    img_shape = (224, 224, 3)

    datagen = dataloader.get_dataloader('data', batch_size=4, shuffle=True)

    model = drn.drn_c_26(pretrained=True)

    generator = LabelGenerator(img_shape, feature_shape)
    result, images = generator.generate(datagen, model)
    print(result.shape)
    print(images.shape)


if __name__ == '__main__':
    main()
