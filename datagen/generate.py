import os
from timeit import default_timer as timer
from typing import Dict, Optional, Tuple

from tqdm.autonotebook import tqdm

from .fea_analysis import FEAnalysis
from .mesh_generator import MeshGenerator
from .utils import find_image_bounds, verify_directory


def generate_data(
    data_dir: str = "data/",
    image_size: int = 512,
    num_plates: int = 1,
    start_plate: Optional[int] = None,
    conditions_per_plate: int = 4,
    mesh_size: float = 1e-2,
    num_polygons_range: Tuple[int, int] = (1, 3),
    points_per_polygon_range: Tuple[int, int] = (3, 8),
    holes_per_polygon_range: Tuple[int, int] = (0, 3),
    points_per_hole_range: Tuple[int, int] = (3, 4),
    num_regions: Tuple[int, int] = (1, 5),
    save_displacement: bool = True,
    save_strain: bool = False,
    save_stress: bool = False,
    num_steps_per_condition: int = 11,
    save_meshes: bool = False,
):
    assert num_steps_per_condition > 1, "Must have at least 2 steps per condition."

    verify_directory(data_dir)

    generator = MeshGenerator(
        num_polygons_range=num_polygons_range,
        points_per_polygon_range=points_per_polygon_range,
        holes_per_polygon_range=holes_per_polygon_range,
        points_per_hole_range=points_per_hole_range,
        num_regions=num_regions,
    )

    assert num_plates >= 1
    assert conditions_per_plate >= 1

    if start_plate is not None:
        assert start_plate < num_plates and start_plate >= 0

    plate_index = (start_plate - 1) if (start_plate is not None) else 0
    plate_image_size, plate_bounds = None, None

    plate_progress_bar = tqdm(total=num_plates, colour="green")
    plate_progress_bar.update(plate_index)
    total_time = 0
    while plate_index < num_plates:
        try:
            geometry = generator.generate_geometry()
        except:
            continue

        # print("PLATE INDEX {}".format(plate_index + 1), "\n")

        geometry = generator.normalize_geometry(geometry)

        polygons_ptags, polygons_ltag_ptags = generator.generate_mesh(
            geometry,
            os.path.join(data_dir, "part"),
            mesh_size=mesh_size,
            view_mesh=False,
        )

        conditions = generator.sample_conditions(
            polygons_ptags, polygons_ltag_ptags, num_conditions=conditions_per_plate
        )

        # print("NUM_CONDITIONS:", len(conditions))

        plate_dir = os.path.join(data_dir, str(plate_index + 1))

        verify_directory(plate_dir)
        for condition_index in range(len(conditions)):
            # print("--- CONDITION INDEX {}".format(condition_index + 1), "\n")
            condition_dir = os.path.join(plate_dir, str(condition_index + 1))
            verify_directory(condition_dir)

            analyzer = FEAnalysis(
                filename="part.mesh",
                data_dir=data_dir,
                condition_dir=condition_dir,
                force_vertex_tags_magnitudes=conditions[condition_index][
                    "point_forces"
                ],
                force_edges_tags_magnitudes=conditions[condition_index]["edge_forces"],
                constraints_vertex_tags=conditions[condition_index][
                    "point_constraints"
                ],
                constraints_edges_tags=conditions[condition_index]["edge_constraints"],
                material_properties_to_vertices=conditions[condition_index][
                    "material_regions"
                ],
                # youngs_modulus = 210000,
                # poisson_ratio = 0.3,
                num_steps=num_steps_per_condition,
                save_meshes=save_meshes,
            )

            start = timer()
            analyzer.calculate()
            end = timer()
            print("TIME:", end - start)
            total_time += end - start

            if condition_index == 0:
                outline_dir = os.path.join(plate_dir, "outline.png")
                analyzer.save_input_image(outline_dir, outline=True, crop=False)
                left, top, right, bottom = find_image_bounds(outline_dir)
                max_size = max(right - left, bottom - top)
                modified_image_size = round(
                    image_size / (max_size / analyzer.initial_image_size)
                )
                analyzer.update_image_size_or_bounds(image_size=modified_image_size)

                analyzer.save_input_image(outline_dir, outline=True, crop=False)
                left, top, right, bottom = find_image_bounds(outline_dir)
                lbound, ubound = (left, right) if right > bottom else (top, bottom)
                bounds = (lbound, lbound, ubound, ubound)
                analyzer.update_image_size_or_bounds(bounds=bounds)
                plate_image_size, plate_bounds = modified_image_size, bounds
                analyzer.save_input_image(os.path.join(plate_dir, "input.png"))
            else:
                analyzer.update_image_size_or_bounds(
                    image_size=plate_image_size, bounds=plate_bounds
                )

            analyzer.save_region_images(os.path.join(condition_dir, "regions"))
            analyzer.save_output_images(
                os.path.join(condition_dir, "outputs"),
                save_displacement=save_displacement,
                save_strain=save_strain,
                save_stress=save_stress,
            )

        plate_index += 1
        plate_progress_bar.update(1)
        print("PLATE TIME:", total_time)
    plate_progress_bar.close()
    print("TOTAL TIME:", total_time)


if __name__ == "__main__":
    generate_data()
