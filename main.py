from geometry_generator import GeometryGenerator
from fea_analysis import calculate_displacement
from typing import Dict
import os
import random
import math

random_generator = random.Random()
init_plate_index = 112
num_plates = 1000
data_dir = "data2/"
mesh_size = 1e-2
initial_image_size = math.ceil(512/0.685546875)
# common_config = "-2 --color-map binary --no-scalar-bars --no-axes --window-size {},{} --off-screen".format(initial_image_size, initial_image_size)
common_config = "-2 --no-axes --window-size {},{} --off-screen".format(initial_image_size, initial_image_size)

generator = GeometryGenerator(num_polygons_range=(1, 3), points_per_polygon_range=(3, 8), holes_per_polygon_range=(0, 3), points_per_hole_range=(3, 4))
forces = [(10000.0, 0.0), (0.0, 10000.0), (10000.0, 10000.0), (10000.0, -10000.0), (-10000.0, 10000.0), (-10000.0, -10000.0), (-10000.0, 0.0), (0.0, -10000.0)]

plate_index = 0 + init_plate_index
while plate_index < num_plates:
    try:
        geometry = generator.generate_geometry()
    except:
        continue
    
    plate_index += 1

    print("PLATE INDEX {}".format(plate_index), "\n")
    
    geometry = generator.normalize_geometry(geometry)
    
    conditions = generator.generate_mesh(geometry, "part", mesh_size=mesh_size, view_mesh = False)
    print("NUM_CONDITIONS:", len(conditions))
    
    for condition_index in range(len(conditions)):
        print("--- CONDITION INDEX {}".format(condition_index), "\n")
        
        force = random_generator.choice(forces)
        
        calculate_displacement("part.mesh", conditions[condition_index]['forces'], conditions[condition_index]['constraints'], (1000.0, 0.0))
        
        if condition_index == 0:
            filename = "{}_plate.png".format(plate_index)
            os.system("sfepy-view linear_elasticity.vtk -f 1:vs {} -o {}/{}".format(common_config, data_dir, filename))
        
        output_file_config : Dict[str, str] = {
            'displacement_x': "linear_elasticity.vtk -f u:c0",
            'displacement_y': "linear_elasticity.vtk -f u:c1",
            'constraints': "regions.vtk -f Constraints:vs",
            'forces': "regions.vtk -f Forces:vs",
        }

        with open("data/plate_forces.txt", "a+") as f:
            f.write("{}, {}, {}\n".format(plate_index, force[0], force[1]))

        for type, config in output_file_config.items():
            filename = "{}_{}_{}.png".format(plate_index, condition_index, type)
            os.system("sfepy-view {} {} -o {}/{}".format(config, common_config, data_dir, filename))