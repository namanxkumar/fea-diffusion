from .mesh_generator import MeshGenerator
from .fea_analysis import FEAnalysis
from .utils import verify_directory, find_image_bounds
from typing import Dict
import os

num_plates = 10
data_dir = "data/"
image_size = 512
verify_directory(data_dir)

mesh_size = 1e-2
generator = MeshGenerator(num_polygons_range=(1, 3), points_per_polygon_range=(3, 8), holes_per_polygon_range=(0, 3), points_per_hole_range=(3, 4))

plate_index = 0
while plate_index < num_plates:
    try:
        geometry = generator.generate_geometry()
    except:
        continue

    print("PLATE INDEX {}".format(plate_index + 1), "\n")
    
    geometry = generator.normalize_geometry(geometry)
    
    polygons_ptags, polygons_ltag_ptags = generator.generate_mesh(geometry, "part", mesh_size=mesh_size, view_mesh = False)

    conditions = generator.sample_conditions(polygons_ptags, polygons_ltag_ptags, num_conditions=4)

    print("NUM_CONDITIONS:", len(conditions))
    
    plate_dir = os.path.join(data_dir, str(plate_index + 1))

    verify_directory(plate_dir)
    plate_image_size = None
    plate_bounds = None

    for condition_index in range(len(conditions)):
        print("--- CONDITION INDEX {}".format(condition_index + 1), "\n")

        analyzer = FEAnalysis('part.mesh', conditions[condition_index]['point_forces'], conditions[condition_index]['edge_forces'], conditions[condition_index]['point_constraints'], conditions[condition_index]['edge_constraints'], youngs_modulus=210000, poisson_ratio=0.3)

        analyzer.calculate()

        condition_dir = os.path.join(plate_dir, str(condition_index + 1))

        verify_directory(condition_dir)

        if condition_index == 0:
            print("Initial Image Size:", analyzer.initial_image_size)
            analyzer.save_input_image(os.path.join(plate_dir, "outline.png"), outline=True, crop=False)

            left, top, right, bottom = find_image_bounds(os.path.join(plate_dir, "outline.png"))

            max_size = max(right - left, bottom - top)
            modified_image_size = int(image_size / (max_size / analyzer.initial_image_size))
            print("Modified Image Size:", modified_image_size)
            
            analyzer.update_image_size_or_bounds(image_size=modified_image_size)

            analyzer.save_input_image(os.path.join(plate_dir, "outline.png"), outline=True, crop=False)

            left, top, right, bottom = find_image_bounds(os.path.join(plate_dir, "outline.png"))
            lbound, ubound = (left, right) if right > bottom else (top, bottom)
            bounds = (lbound, lbound, ubound, ubound)
            print("Bounds:", bounds)
            analyzer.update_image_size_or_bounds(bounds=bounds)
            plate_image_size = modified_image_size
            plate_bounds = bounds
            analyzer.save_input_image(os.path.join(plate_dir, "input.png"))
        else:
            analyzer.update_image_size_or_bounds(image_size=plate_image_size, bounds=plate_bounds)

        analyzer.save_region_images(os.path.join(condition_dir, "regions"))
        analyzer.save_output_images(os.path.join(condition_dir, "outputs"), save_displacement=True, save_strain=False, save_stress=False)

    plate_index += 1