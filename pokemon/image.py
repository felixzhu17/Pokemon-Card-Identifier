import os
import matplotlib.pyplot as plt
from PIL import Image
import json
from torchvision import transforms
from torch.utils.data import Dataset
import torch
from torch.nn import MSELoss
import numpy as np
from albumentations import (
    Compose, RandomBrightnessContrast, ShiftScaleRotate, 
    KeypointParams, ISONoise, HueSaturationValue, ImageCompression
)


def augment_data(pokemon_data, max_attempts=1000):
    for _ in range(max_attempts):
        transformed_pokemon_image = _try_augment_data(pokemon_data)
        if transformed_pokemon_image is not None:
            return transformed_pokemon_image

def _try_augment_data(pokemon_data):
    transform = Compose([
        RandomBrightnessContrast(),
        ShiftScaleRotate(shift_limit = (-0.1, 0.1), scale_limit = (-0.2, 0.1), rotate_limit = (-45, 45)),
        ISONoise(),
        HueSaturationValue(),
        ImageCompression()
    ], keypoint_params=KeypointParams(format='yx')) 
    transformed = transform(image=np.array(pokemon_data.original_image), keypoints=pokemon_data.original_annotation)
    transformed_image = transformed["image"]
    transformed_annotations = transformed["keypoints"]
    if len(transformed_annotations) != len(pokemon_data.original_annotation):
        return None
    else:
        return PokemonImage(pokemon_data.image_name, Image.fromarray(transformed_image), transformed_annotations)


class PokemonData(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

def extract_corners(annotations, name):
    corners = []
    assert len(annotations) == 8, name
    for d in annotations:
        assert len(d["shape_attributes"]["all_points_x"]) == 2, name 
        assert len(d["shape_attributes"]["all_points_y"]) == 2, name
        for x, y in zip(
            d["shape_attributes"]["all_points_x"], d["shape_attributes"]["all_points_y"]
        ):
            corners.append((y, x))
    assert len(corners) == 16, name
    return corners


def load_image_data(image_dir, annotation_path):
    # Get a list of all jpg files in the folder
    jpg_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    with open(annotation_path, "r") as f:
        annotations = json.load(f)
    annotations = list(annotations.values())
    annotations = {i["filename"]: extract_corners(i["regions"], i["filename"]) for i in annotations}
    images = {
        img_path: Image.open(os.path.join(image_dir, img_path))
        for img_path in jpg_files
    }
    data = [PokemonImage(image, images.get(image), annotations.get(image)) for image in images]
    return data

def load_test_image_data(image_dir):
    jpg_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    images = {
        img_path: Image.open(os.path.join(image_dir, img_path))
        for img_path in jpg_files
    }
    data = [PokemonImage(image, images.get(image)) for image in images]
    return data

class PokemonImage:
    def __init__(self, image_name, image, annotation = None, resized_image_size=(224, 224)):
        self.image_name = image_name
        self.original_image = image
        self.original_image_size = image.size[:2]
        self.resized_image_size = resized_image_size
        self.resized_image = self.resize_image()
        
        if annotation:
            self.original_annotation = annotation
            self.original_line_points = self.get_line_points(self.original_annotation)
            self.original_side_midpoints, self.original_centering = self.get_centering(self.original_line_points)

            self.resized_annotation = self.resize_annotations()
            self.resized_line_points = self.get_line_points(self.resized_annotation)
            self.resized_side_midpoints, self.resized_centering = self.get_centering(self.resized_line_points)

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
        pred_line_points = self.get_line_points(pred_annotations)
        pred_side_midpoints, pred_centering = self.get_centering(pred_line_points)
        self.plot_prediction(pred_annotations, pred_side_midpoints, pred_centering)

    def plot(self):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(self.original_image)
        if hasattr(self, "original_annotation"):
            self._plot_annotations(axs[0], self.original_annotation)
            self._plot_midpoints(axs[0], self.original_side_midpoints)
            self._print_centering(self.original_centering, "Original")
        axs[0].set_title("Original Image")

        # Resize image and convert to numpy for plotting
        resized_numpy_image = self.resized_image.permute(1, 2, 0).numpy()
        axs[1].imshow(resized_numpy_image)
        if hasattr(self, "resized_annotation"):
            self._plot_annotations(axs[1], self.resized_annotation)
            self._plot_midpoints(axs[1], self.resized_side_midpoints)
            self._print_centering(self.resized_centering, "Resized")
        axs[1].set_title("Resized Image")

        plt.show()

    def plot_prediction(self, pred_annotations, pred_side_midpoints, pred_centering):
        if hasattr(self, "original_annotation"):
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].imshow(self.original_image)
            self._plot_annotations(axs[0], self.original_annotation)
            self._plot_midpoints(axs[0], self.original_side_midpoints)
            self._print_centering(self.original_centering, "Actual")
            axs[0].set_title("Actual")

            axs[1].imshow(self.original_image)
            self._plot_annotations(axs[1], pred_annotations)
            self._plot_midpoints(axs[1], pred_side_midpoints)
            self._print_centering(pred_centering, "Predicted")
            axs[1].set_title("Predicted")

        else: 
            fig, axs = plt.subplots(figsize=(6, 6))
            axs.imshow(self.original_image)
            self._plot_annotations(axs, pred_annotations)
            self._plot_midpoints(axs, pred_side_midpoints)
            self._print_centering(pred_centering, "Predicted")
            axs.set_title("Predicted")

        plt.show()

    def get_line_points(self, annotations):
        pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)]
        labels = ['VL1', 'HB2', 'VR2', 'HT1', 'VL2', 'HB1', 'VR1', 'HT2']

        line_points = {}
        for pair, label in zip(pairs, labels):
            # Retrieve the coordinates
            line_points[label] = (annotations[pair[0]], annotations[pair[1]])
        return line_points

    def get_centering(self, line_points):

        L_width, L_points = self._orthogonal_distance_and_point(self._find_midpoint(*line_points['VL2']), line_points['VL1'])
        R_width, R_points = self._orthogonal_distance_and_point(self._find_midpoint(*line_points['VR2']), line_points['VR1'])
        T_width, T_points = self._orthogonal_distance_and_point(self._find_midpoint(*line_points['HT2']), line_points['HT1'])
        B_width, B_points = self._orthogonal_distance_and_point(self._find_midpoint(*line_points['HB2']), line_points['HB1'])
        
        center_points = {"L": L_points, "R": R_points, "T": T_points, "B": B_points}
        horizontal_centering = L_width/R_width
        vertical_centering = T_width/B_width
        return center_points, (horizontal_centering, vertical_centering)

    def _find_midpoint(self, point_1, point_2):
        y1, x1 = point_1
        y2, x2 = point_2
        return ((y1 + y2) / 2, (x1 + x2) / 2)
        
    def _find_distance(self, point_1, point_2):
        y1, x1 = point_1
        y2, x2 = point_2
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def _print_centering(self, centering, message):
        horizontal_centering, vertical_centering = centering
        print(f"{message} Horizontal Centering: {horizontal_centering}")
        print(f"{message} Vertical Centering: {vertical_centering}")

    def _orthogonal_distance_and_point(self, point, line_points):
        # point: (y, x)
        # line_points: ((y1, x1), (y2, x2))
        y0, x0 = point
        y1, x1 = line_points[0]
        y2, x2 = line_points[1]

        # Calculate the numerator and denominator of the distance formula
        numerator = abs((x2-x1)*y0 - (y2-y1)*x0 + y2*x1 - x2*y1)
        denominator = np.sqrt((x2-x1)**2 + (y2-y1)**2)

        # Calculate the distance
        distance = numerator / denominator

        # Calculate the intersection point
        dx, dy = x2-x1, y2-y1
        t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy)
        x, y = x1 + t * dx, y1 + t * dy

        # return the distance and intersection point
        return distance, (point, (y, x))

    def _plot_annotations(self, axs, annotations):
        for i in range(0, len(annotations), 2):
            y1, x1 = annotations[i]
            y2, x2 = annotations[i+1]
            axs.plot([x1, x2], [y1, y2], "grey", markersize=2)

    def _plot_midpoints(self, axs, midpoints):
        for label, midpoint_list in midpoints.items():
            for midpoint in midpoint_list:
                y, x = midpoint
                axs.plot(x, y, "go", markersize=2) 