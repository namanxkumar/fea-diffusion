from .mesh_generator import MeshGenerator
from .fea_analysis import FEAnalysis
import os
from PIL import Image


def verify_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def create_box_mesh(mesh_size):
    box_generator = MeshGenerator()
    box = box_generator.create_box()
    box_generator.generate_mesh(box, "box", mesh_size=mesh_size, view_mesh=False)


def find_image_bounds(image_path):
    # common_config = "-2 --color-map binary --no-scalar-bars --no-axes --window-size {},{} --off-screen".format(image_size, image_size)
    # os.system("sfepy-view box.mesh -f 1:vs {} --outline -o {}".format(common_config, image_path))
    image = Image.open(image_path)
    # find the bounding box of the geometry by finding the first non-white pixel
    image_data = image.load()
    left = 0
    right = image.size[0]
    top = 0
    bottom = image.size[1]
    for x in range(image.size[0]):
        for y in range(image.size[1]):
            if image_data[x, y] != (255, 255, 255):
                left = x
                break
        if left != 0:
            break
    for x in range(image.size[0] - 1, -1, -1):
        for y in range(image.size[1]):
            if image_data[x, y] != (255, 255, 255):
                right = x
                break
        if right != image.size[0]:
            break
    for y in range(image.size[1]):
        for x in range(image.size[0]):
            if image_data[x, y] != (255, 255, 255):
                top = y
                break
        if top != 0:
            break
    for y in range(image.size[1] - 1, -1, -1):
        for x in range(image.size[0]):
            if image_data[x, y] != (255, 255, 255):
                bottom = y
                break
        if bottom != image.size[1]:
            break
    return left, top, right, bottom


def find_box_bounds(required_image_size):
    analyzer = FEAnalysis("box.mesh", [], [], [], [])

    analyzer.save_input_image(
        "box.png", input_filepath="box.mesh", outline=True, crop=False
    )
    left, top, right, bottom = find_image_bounds("box.png")
    max_size = max(right - left, bottom - top)
    modified_image_size = int(
        required_image_size / (max_size / analyzer.initial_image_size)
    )
    analyzer.update_image_size_or_bounds(image_size=modified_image_size)

    analyzer.save_input_image(
        "box.png", input_filepath="box.mesh", outline=True, crop=False
    )
    left, top, right, bottom = find_image_bounds("box.png")
    lbound, ubound = (left, right) if right > bottom else (top, bottom)
    bounds = (lbound, lbound, ubound, ubound)

    return modified_image_size, bounds
