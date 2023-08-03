import os
import matplotlib.pyplot as plt
from PIL import Image
import json
from torchvision import transforms
from torch.utils.data import DataLoader
import torch


class PokemonData(DataLoader):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            indices = list(range(*idx.indices(len(self))))
            images = torch.empty((len(indices), *self.dataset[0].resized_image.shape))
            annotations = torch.empty((len(indices), *torch.tensor(self.dataset[0].resized_annotation).flatten().shape))
            for i, index in enumerate(indices):
                images[i] = self.dataset[index].resized_image.clone().detach()
                annotations[i] = torch.tensor(self.dataset[index].resized_annotation).flatten()
            return images, annotations
        else:
            return (
                self.dataset[idx].resized_image.clone().detach(),
                self.dataset[idx].resized_annotation.clone().detach().flatten()
            )

def extract_corners(annotations):
    corners = []
    for d in annotations:
        for x, y in zip(
            d["shape_attributes"]["all_points_x"], d["shape_attributes"]["all_points_y"]
        ):
            corners.append((y, x))
    return corners


def load_image_data(image_dir, annotation_path):
    # Get a list of all jpg files in the folder
    jpg_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    with open(annotation_path, "r") as f:
        annotations = json.load(f)
    annotations = list(annotations.values())
    annotations = {i["filename"]: extract_corners(i["regions"]) for i in annotations}
    images = {
        img_path: Image.open(os.path.join(image_dir, img_path))
        for img_path in jpg_files
    }
    data = [PokemonImage(images.get(image), annotations.get(image)) for image in images]
    return data


class PokemonImage:
    def __init__(self, image, annotation, resized_image_size=(224, 224)):
        self.original_image = image
        self.original_annotation = annotation
        self.original_image_size = image.size[:2]
        self.resized_image_size = resized_image_size
        self.resized_image = self.resize_image()
        self.resized_annotation = self.resize_annotations()

    def resize_image(self):
        transform = transforms.Compose(
            [transforms.Resize(self.resized_image_size), transforms.ToTensor()]
        )
        return transform(self.original_image)

    def resize_annotations(self):
        scale_x = self.resized_image_size[0] / self.original_image_size[0]
        scale_y = self.resized_image_size[1] / self.original_image_size[1]
        resized_annotations = [
            (int(y * scale_y), int(x * scale_x)) for (y, x) in self.original_annotation
        ]
        return resized_annotations

    def undo_resize_image(self, resized_image_tensor):
        # Convert the tensor back to a PIL Image and resize
        transform = transforms.Compose(
            [transforms.ToPILImage(), transforms.Resize(self.original_image_size)]
        )
        return transform(resized_image_tensor)

    def undo_resize_annotations(self, resized_annotations):
        scale_x = self.original_image_size[0] / self.resized_image_size[0]
        scale_y = self.original_image_size[1] / self.resized_image_size[1]
        original_annotations = [
            (int(y * scale_y), int(x * scale_x)) for (y, x) in resized_annotations
        ]
        return original_annotations

    def plot(self):
        fig, axs = plt.subplots(
            1, 2, figsize=(12, 6)
        )  # Create 1 row, 2 columns of subplots

        # Original Image
        # Convert the tensor image to numpy and permute it for correct plotting
        # Display the original image
        axs[0].imshow(self.original_image)
        # Plot the original annotations
        for (y, x) in self.original_annotation:
            axs[0].plot(x, y, "ro")
        axs[0].set_title("Original Image")

        # Resized Image
        # Convert the tensor image to numpy and permute it for correct plotting
        resized_numpy_image = self.resized_image.permute(1, 2, 0).numpy()
        # Display the resized image
        axs[1].imshow(resized_numpy_image)
        # Plot the resized annotations
        for (y, x) in self.resized_annotation:
            axs[1].plot(x, y, "ro")
        axs[1].set_title("Resized Image")

        plt.show()