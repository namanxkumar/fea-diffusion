from geometry_generator import GeometryGenerator
import os
from PIL import Image

def create_box_mesh(mesh_size):
    box_generator = GeometryGenerator(num_polygons_range=(1, 3), points_per_polygon_range=(3, 8), holes_per_polygon_range=(0, 3), points_per_hole_range=(3, 4))
    box = box_generator.create_box()
    box_generator.generate_mesh(box, "box", mesh_size=mesh_size, view_mesh = False)

def find_image_bounds(image_path, image_size):
    common_config = "-2 --color-map binary --no-scalar-bars --no-axes --window-size {},{} --off-screen".format(image_size, image_size)
    os.system("sfepy-view box.mesh -f 1:vs {} --outline -o {}".format(common_config, image_path))
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

def crop_image(image_path, bounds):
    image = Image.open(image_path)
    image = image.crop(bounds)
    image.save(image_path)
