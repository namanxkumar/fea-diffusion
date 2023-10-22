from .mesh_generator import MeshGenerator
from .fea_analysis import FEAnalysis
from .utils import verify_directory, find_image_bounds
from typing import Dict
import os
from tqdm import tqdm

def generate_data(data_dir = "data/", image_size = 512, num_plates = 1, conditions_per_plate = 4, mesh_size = 1e-2, num_polygons_range=(1, 3), points_per_polygon_range=(3, 8), holes_per_polygon_range=(0, 3), points_per_hole_range=(3, 4), save_displacement=True, save_strain=False, save_stress=False):
    verify_directory(data_dir)

    generator = MeshGenerator(num_polygons_range=num_polygons_range, points_per_polygon_range=points_per_polygon_range, holes_per_polygon_range=holes_per_polygon_range, points_per_hole_range=points_per_hole_range)

    plate_index = 0
    plate_image_size, plate_bounds = None, None

    plate_progress_bar = tqdm(total=num_plates, colour="green")
    while plate_index < num_plates:
        try:
            geometry = generator.generate_geometry()
        except:
            continue

        print("PLATE INDEX {}".format(plate_index + 1), "\n")
        
        geometry = generator.normalize_geometry(geometry)
        
        polygons_ptags, polygons_ltag_ptags = generator.generate_mesh(geometry, "part", mesh_size=mesh_size, view_mesh = False)

        conditions = generator.sample_conditions(polygons_ptags, polygons_ltag_ptags, num_conditions=conditions_per_plate)

        print("NUM_CONDITIONS:", len(conditions))
        
        plate_dir = os.path.join(data_dir, str(plate_index + 1))

        verify_directory(plate_dir)
        for condition_index in tqdm(range(len(conditions)), colour="blue"):
            # print("--- CONDITION INDEX {}".format(condition_index + 1), "\n")

            analyzer = FEAnalysis('part.mesh', conditions[condition_index]['point_forces'], conditions[condition_index]['edge_forces'], conditions[condition_index]['point_constraints'], conditions[condition_index]['edge_constraints'], youngs_modulus=210000, poisson_ratio=0.3)

            analyzer.calculate()

            condition_dir = os.path.join(plate_dir, str(condition_index + 1))

            verify_directory(condition_dir)

            if condition_index == 0:
                outline_dir = os.path.join(plate_dir, "outline.png")
                analyzer.save_input_image(outline_dir, outline=True, crop=False)
                left, top, right, bottom = find_image_bounds(outline_dir)
                max_size = max(right - left, bottom - top)
                modified_image_size = round(image_size / (max_size / analyzer.initial_image_size))
                analyzer.update_image_size_or_bounds(image_size=modified_image_size)
                
                analyzer.save_input_image(outline_dir, outline=True, crop=False)
                left, top, right, bottom = find_image_bounds(outline_dir)
                lbound, ubound = (left, right) if right > bottom else (top, bottom)
                bounds = (lbound, lbound, ubound, ubound)
                analyzer.update_image_size_or_bounds(bounds=bounds)
                plate_image_size, plate_bounds = modified_image_size, bounds
                analyzer.save_input_image(os.path.join(plate_dir, "input.png"))
            else:
                analyzer.update_image_size_or_bounds(image_size=plate_image_size, bounds=plate_bounds)

            analyzer.save_region_images(os.path.join(condition_dir, "regions"))
            analyzer.save_output_images(os.path.join(condition_dir, "outputs"), save_displacement=save_displacement, save_strain=save_strain, save_stress=save_stress)

        plate_index += 1
        plate_progress_bar.update(1)
    plate_progress_bar.close()

if __name__ == "__main__":
    generate_data()