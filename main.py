from geometry_generator import GeometryGenerator
from fea_analysis import calculate_displacement
from typing import Dict
import os

data_dir = "data/"

generator = GeometryGenerator(num_polygons_range=(1, 3), points_per_polygon_range=(3, 8), holes_per_polygon_range=(0, 3), points_per_hole_range=(3, 4))

# generator.plot_geometry(geometry)

# geometry = generator.create_box()
num_plates = 1
plate_index = 0
while plate_index < num_plates:
    print("PLATE INDEX {}".format(plate_index), "\n")
    try:
        geometry = generator.generate_geometry()
    except:
        continue
    plate_index += 1
    geometry = generator.normalize_geometry(geometry)
    conditions = generator.generate_mesh(geometry, "part", mesh_size= 1e-2, view_mesh = False)
    print("NUM_CONDITIONS:", len(conditions))
    for condition_index in range(len(conditions)):
        print("--- CONDITION INDEX {}".format(condition_index), "\n")
        calculate_displacement("part.mesh", conditions[condition_index]['forces'], conditions[condition_index]['constraints'], (10000.0, 0.0))
        output_file_config : Dict[str, str] = {
            'plate': "-f 1:vs",
            'displacement_x': "-f u:c0",
            'displacement_y': "-f u:c1",
        }
        common_config = "-2 --color-map Greys --no-scalar-bars --no-axes --window-size 5000,5000 --off-screen"
        for type, config in output_file_config.items():
            os.system("sfepy-view linear_elasticity.vtk {} {} -o {}/{}_{}_{}.png".format(config, common_config, data_dir, plate_index, condition_index, type))