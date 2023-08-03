import os
import matplotlib.pyplot as plt
from PIL import Image
import json
from torchvision import transforms
from torch.utils.data import Dataset
import torch
from torch.nn import MSELoss

class PokemonData(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


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
    def __init__(self, image, annotation = None, resized_image_size=(224, 224)):
        self.original_image = image
        self.original_image_size = image.size[:2]
        self.resized_image_size = resized_image_size
        self.resized_image = self.resize_image()
        if annotation:
            self.original_annotation = annotation
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

    def predict_annotations(self, model):
        pred_annotations = model(self.resized_image.float().unsqueeze(0)).detach().cpu()

        if hasattr(self, "resized_annotation"):
            actual_annotations = torch.tensor(self.resized_annotation).flatten().float()
            print("Loss: ", MSELoss()(pred_annotations, actual_annotations).item())

        pred_annotations = pred_annotations.reshape(-1, 2).tolist()
        pred_annotations = self.undo_resize_annotations(pred_annotations)
        self.plot_prediction(pred_annotations)


    def plot(self):
        fig, axs = plt.subplots(
            1, 2, figsize=(12, 6)
        )  # Create 1 row, 2 columns of subplots

        # Original Image
        # Convert the tensor image to numpy and permute it for correct plotting
        # Display the original image
        axs[0].imshow(self.original_image)
        # Plot the original annotations
        for y, x in self.original_annotation:
            axs[0].plot(x, y, "ro")
        axs[0].set_title("Original Image", markersize=2)

        # Resized Image
        # Convert the tensor image to numpy and permute it for correct plotting
        resized_numpy_image = self.resized_image.permute(1, 2, 0).numpy()
        # Display the resized image
        axs[1].imshow(resized_numpy_image)
        # Plot the resized annotations
        for y, x in self.resized_annotation:
            axs[1].plot(x, y, "ro", markersize=2)
        axs[1].set_title("Resized Image")

        plt.show()

    def plot_prediction(self, predicted_annotations):
        fig, axs = plt.subplots(
            1, 2, figsize=(12, 6)
        )  # Create 1 row, 2 columns of subplots

        # Display the resized image
        axs[0].imshow(self.original_image)
        # Plot the resized annotations
        for y, x in self.original_annotation:
            axs[0].plot(x, y, "ro", markersize=2)
        axs[0].set_title("Actual")

        # Display the resized image
        axs[1].imshow(self.original_image)
        # Plot the resized annotations
        for y, x in predicted_annotations:
            axs[1].plot(x, y, "ro", markersize=2)
        axs[1].set_title("Predicted")

        plt.show()
