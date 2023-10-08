import shapely.geometry
from shapely.plotting import plot_polygon
import shapely.affinity
from shapely.ops import unary_union
import random
import matplotlib.pyplot as plt
import gmsh
import sys
from collections import OrderedDict

class GeometryGenerator():
    def __init__(self, num_polygons_range, points_per_polygon_range, holes_per_polygon_range, points_per_hole_range, random_seed = None):
        self.num_polygons_range = num_polygons_range
        self.points_per_polygon_range = points_per_polygon_range
        self.holes_per_polygon_range = holes_per_polygon_range
        self.points_per_hole_range = points_per_hole_range

        self.random = random.Random(random_seed)

    def random_float(self):
        return float(self.random.randint(0, 1000)) / 1000
    
    def random_coordinates(self, num_coordinates, bounds = None):
        if bounds is None:
            bounds = (0, 0, 1, 1)
        return [(bounds[0] + self.random_float() * (bounds[2] - bounds[0]), bounds[1] + self.random_float() * (bounds[3] - bounds[1])) for i in range(num_coordinates)]
    
    def random_coordinates_in_polygon(self, polygon : shapely.geometry.Polygon, num_coordinates):
        coordinates = []
        while len(coordinates) < num_coordinates:
            sampled_coordinate = polygon.point_on_surface()
            if sampled_coordinate not in coordinates:
                coordinates.append(sampled_coordinate)

    def random_polygon(self, exterior_polygon : shapely.geometry.Polygon = None, generate_holes : bool = True) -> shapely.geometry.Polygon:
        if exterior_polygon is None:
            num_points = self.random.randint(*self.points_per_polygon_range)
            
            bounds_for_diversity = [[0.5, 0, 1, 1], [0, 0, 0.5, 1], [0, 0.5, 1, 1], [0, 0, 1, 0.5]]
            self.random.shuffle(bounds_for_diversity)
            
            outer = self.random_coordinates(num_points//3, bounds_for_diversity[0]) + self.random_coordinates(num_points//3, bounds_for_diversity[1]) + self.random_coordinates(num_points - 2*num_points//3, bounds_for_diversity[2])
            
            exterior_polygon: shapely.geometry.Polygon = shapely.geometry.MultiPoint(outer).convex_hull
        
        if not generate_holes:
            return exterior_polygon
        
        holes = []
        for _ in range(self.random.randint(*self.holes_per_polygon_range)):
            num_points = self.random.randint(*self.points_per_hole_range)
            while True:
                sampled_interior_coordinates = self.random_coordinates(num_points, exterior_polygon.bounds)
                # sampled_interior_coordinates = self.random_coordinates_in_polygon(exterior_polygon, num_points)
                
                interior_polygon: shapely.geometry.Polygon = shapely.geometry.Polygon(sampled_interior_coordinates).convex_hull

                if exterior_polygon.contains_properly(interior_polygon) and len([1 for hole in holes if interior_polygon.intersects(shapely.geometry.LinearRing(hole))]) == 0:
                    holes.append(interior_polygon.exterior.coords[::-1])
                    break
        return shapely.geometry.Polygon(exterior_polygon.exterior.coords, holes)

    def random_multi_polygon(self) -> shapely.geometry.MultiPolygon:
        num_polygons = self.random.randint(*self.num_polygons_range)
        return shapely.geometry.MultiPolygon([self.random_polygon(generate_holes=False) for _ in range(num_polygons)])

    def generate_geometry(self):
        geometry = self.random_multi_polygon()
        geometry : shapely.geometry.Polygon = unary_union(geometry)
        geometry = self.random_polygon(exterior_polygon=geometry)
        return geometry
    
    @staticmethod
    def create_box():
        return shapely.geometry.Polygon(((0, 0), (1, 0), (1, 1), (0, 1)))

    @staticmethod
    def normalize_geometry(geometry):
        bounds = geometry.bounds
        geometry = shapely.affinity.translate(geometry, (0 - bounds[0]), (0 - bounds[1]))

        scale = 1.0 / max(bounds[2] - bounds[0], bounds[3] - bounds[1])
        geometry = shapely.affinity.scale(geometry, scale, scale, origin=(0, 0, 0))

        return geometry
    
    @staticmethod
    def plot_geometry(geometry):
        fig = plt.figure(1, figsize=(5,5), dpi=90)
        ax = fig.add_subplot(111)
        plot_polygon(geometry, ax, facecolor=(0.9, 0.9, 0.9), edgecolor=(0, 0, 0))
        plt.show()

    def generate_mesh(self, geometry, name, num_conditions_sets = 4, mesh_size = 0.1, view_mesh = False, filetype = 'mesh'):
        gmsh.initialize() # Initialize the Gmsh API

        # Get the coordinates of the external and internal polygons
        internal_polygons_coordinates = []
        for polygon in geometry.interiors:
            internal_polygons_coordinates.append(polygon.coords[:-1]) # The last point is the same as the first point
        external_polygon_coordinates = geometry.exterior.coords[:-1] # The last point is the same as the first point
        
        # Create the Gmsh geometry

        internal_polygons = []
        internal_polygons_lines = OrderedDict()
        # Create the internal polygons
        for coordinates in internal_polygons_coordinates:
            internal_polygon_points = []
            for point in coordinates:
                internal_polygon_points.append(gmsh.model.geo.add_point(point[0], point[1], 0.0, mesh_size))
            
            internal_polygons_lines = OrderedDict()
            for i in range(len(internal_polygon_points) - 1):
                internal_polygons_lines[gmsh.model.geo.add_line(internal_polygon_points[i], internal_polygon_points[i+1])] = (internal_polygon_points[i], internal_polygon_points[i+1])
            internal_polygons_lines[gmsh.model.geo.add_line(internal_polygon_points[-1], internal_polygon_points[0])] = (internal_polygon_points[-1], internal_polygon_points[0]) # Close the polygon
            
            internal_polygons.append(gmsh.model.geo.add_curve_loop(list(internal_polygons_lines.keys())))
        
        # Create the external polygon
        external_polygon_points = []
        for point in external_polygon_coordinates:
            external_polygon_points.append(gmsh.model.geo.add_point(point[0], point[1], 0.0, mesh_size))
        
        external_polygon_lines = OrderedDict()
        for i in range(len(external_polygon_points) - 1):
            external_polygon_lines[gmsh.model.geo.add_line(external_polygon_points[i], external_polygon_points[i+1])] = (external_polygon_points[i], external_polygon_points[i+1])
        external_polygon_lines[gmsh.model.geo.add_line(external_polygon_points[-1], external_polygon_points[0])] = (external_polygon_points[-1], external_polygon_points[0]) # Close the polygon
        
        external_polygon = gmsh.model.geo.add_curve_loop(list(external_polygon_lines.keys()))

        surface = gmsh.model.geo.add_plane_surface([external_polygon, *internal_polygons]) # Create the surface

        # Create the constraints
        conditions = []
        # Set the physical groups
        
        # Sample Points and Lines
        # sampled_points = self.random.sample(external_polygon_points, 1)

        # for point in sampled_points:
        # gmsh.model.addPhysicalGroup(0, sampled_points, name="force")
        while(len(conditions) < num_conditions_sets):
            sampled_points = self.random.sample(external_polygon_points, 1)
            sampled_internal_lines_points = self.random.sample(list(internal_polygons_lines.values()), self.random.randint(0, len(internal_polygons_lines) - 1)) if len(internal_polygons_lines) > 0 else []
            sampled_external_lines_points = self.random.sample(list(external_polygon_lines.values()), self.random.randint(1, len(external_polygon_lines) - 1))

            conditions_dict = {
                "forces": sampled_points,
                "constraints": sampled_external_lines_points + sampled_internal_lines_points
            }
            if conditions_dict not in conditions:
                conditions.append(conditions_dict)
            

        # sampled_internal_lines_points = self.random.sample(list(internal_polygons_lines.values()), self.random.randint(0, len(internal_polygons_lines) - 1)) if len(internal_polygons_lines) > 0 else []
        # sampled_external_lines_points = self.random.sample(list(external_polygon_lines.values()), self.random.randint(1, len(external_polygon_lines) - 1))
        # sampled_external_lines_points = list(external_polygon_lines.values())[:2]
        
        # gmsh.model.addPhysicalGroup(1, sampled_internal_lines + sampled_external_lines, name="constraints")
        
        gmsh.model.addPhysicalGroup(2, [surface], name="surface") # Add the surface to a physical group
        
        gmsh.model.geo.synchronize() # Synchronize the Gmsh model
        
        gmsh.model.mesh.generate() # Generate the mesh

        gmsh.write("{}.geo_unrolled".format(name)) # Write the geometry to a file
        
        gmsh.write("{}.{}".format(name, filetype)) # Write the mesh to a file
        # gmsh.write("{}.msh".format(name)) # Write the mesh to a file
        
        if view_mesh:
            if 'close' not in sys.argv:
                gmsh.fltk.run()
        # Close the Gmsh API
        gmsh.finalize()

        return conditions