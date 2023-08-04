import math
import torch
import torchvision.transforms.functional as F
import numpy as np
from pokemon.image import PokemonImage
import random
import torchvision.transforms as transforms


def generate_random_transform():
    # Define the parameters for the affine transformation
    angle = np.random.uniform(-15, 15)
    scale = np.random.uniform(0.8, 1.1)
    translate = (np.random.randint(-5, 5), np.random.randint(-5, 5))
    shear = (np.random.randint(-2, 2), np.random.randint(-2, 2))
    return angle, scale, translate, shear


def apply_affine_transform(pokemon_image, angle, scale, translate, shear):
    img_size = pokemon_image.original_image.size
    center = [img_size[0] * 0.5, img_size[1] * 0.5]
    color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
    jittered_image = color_jitter(pokemon_image.original_image)
    matrix = get_transform_matrix(center, angle, translate, scale, shear)
    transformed_image = F.affine(
        jittered_image, angle, translate, scale, shear, fill=random_color()
    )

    # Define the transformation matrix
    transformation_matrix = torch.tensor([matrix + [0, 0, 1]]).reshape(3, 3)

    # Apply the transformation to the annotations
    annotation_tensors = torch.tensor(pokemon_image.original_annotation).T
    homogeneous_annotations = torch.cat(
        (annotation_tensors, torch.ones((1, annotation_tensors.shape[1]))), dim=0
    )

    cartesian_to_image_matrix = torch.tensor(
        [[0, -1, img_size[1]], [1, 0, 0], [0, 0, 1]]
    ).float()
    image_to_cartesian_matrix = torch.tensor(
        [[0, 1, 0], [-1, 0, img_size[1]], [0, 0, 1]]
    ).float()

    image_transformation_matrix = torch.mm(
        cartesian_to_image_matrix,
        torch.mm(transformation_matrix, image_to_cartesian_matrix),
    )
    transformed_annotations = torch.mm(
        image_transformation_matrix, homogeneous_annotations
    )
    transformed_annotations = transformed_annotations[:-1].numpy().T
    if not check_annotations(transformed_annotations):
        return None
    else:
        transformed_annotations = [tuple(row) for row in transformed_annotations]
        return PokemonImage(pokemon_image.image_name, transformed_image, transformed_annotations)


def affine_transform_pokemon_image(pokemon_image, max_attempts=1000):
    for _ in range(max_attempts):
        angle, scale, translate, shear = generate_random_transform()
        transformed_pokemon_image = apply_affine_transform(
            pokemon_image, angle, scale, translate, shear
        )
        if transformed_pokemon_image is not None:
            return transformed_pokemon_image
    return None


def get_transform_matrix(center, angle, translate, scale, shear):
    # Helper method to compute inverse matrix for affine transformation

    # As it is explained in PIL.Image.rotate
    # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RSS is rotation with scale and shear matrix
    #       RSS(a, s, (sx, sy)) =
    #       = R(a) * S(s) * SHy(sy) * SHx(sx)
    #       = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(x)/cos(y) - sin(a)), 0 ]
    #         [ s*sin(a + sy)/cos(sy), s*(-sin(a - sy)*tan(x)/cos(y) + cos(a)), 0 ]
    #         [ 0                    , 0                                      , 1 ]
    #
    # where R is a rotation matrix, S is a scaling matrix, and SHx and SHy are the shears:
    # SHx(s) = [1, -tan(s)] and SHy(s) = [1      , 0]
    #          [0, 1      ]              [-tan(s), 1]
    #
    # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

    rot = math.radians(angle)
    sx, sy = [math.radians(s) for s in shear]

    cx, cy = center
    tx, ty = translate

    # RSS without scaling
    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

    # Inverted rotation matrix with scale and shear
    # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
    matrix = [d, -b, 0.0, -c, a, 0.0]
    matrix = [x * scale for x in matrix]

    # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
    matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
    matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)

    # Apply center translation: C * RSS^-1 * C^-1 * T^-1
    matrix[2] += cx
    matrix[5] += cy

    return matrix


def check_annotations(arr, img_size):
    if np.any(arr < 0):
        return False
    if np.any(arr[:, 0] > img_size[1]):
        return False
    if np.any(arr[:, 1] > img_size[0]):
        return False
    return True

def random_color():
    red = random.randint(0, 255)
    green = random.randint(0, 255)
    blue = random.randint(0, 255)
    return (red, green, blue)

