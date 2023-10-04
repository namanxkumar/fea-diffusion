from geometry_generator import GeometryGenerator

generator = GeometryGenerator(num_polygons_range=(1, 3), points_per_polygon_range=(3, 8), holes_per_polygon_range=(0, 3), points_per_hole_range=(3, 4))

geometry = generator.generate_geometry()
geometry = generator.normalize_geometry(geometry)
generator.plot_geometry(geometry)
generator.generate_mesh(geometry, "part1")