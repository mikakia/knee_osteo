import torch

from datafunctions.dataset import KneeOsteoDataset
from models.cnn import CNN


def main():
    dataset = KneeOsteoDataset()
    dataset.summary()

    model = CNN(num_classes=len(dataset.classes))

    image, _label = dataset[0]
    output = model(image.unsqueeze(0))
    print("Output shape:", output.shape)
    print("Predicted class:", torch.argmax(output, dim=1).item())

if __name__ == "__main__":
    main()