import shapely.geometry
from shapely.plotting import plot_polygon
import shapely.affinity
from shapely.ops import unary_union
import random
import matplotlib.pyplot as plt
import gmsh
import sys
from collections import OrderedDict
from typing import List, Tuple, OrderedDict, Dict
from copy import deepcopy

class MeshGenerator():
    def __init__(self, num_polygons_range=(1, 3), points_per_polygon_range=(3, 8), holes_per_polygon_range=(0, 3), points_per_hole_range=(3, 4), random_seed = None):
        self.num_polygons_range = num_polygons_range
        self.points_per_polygon_range = points_per_polygon_range
        self.holes_per_polygon_range = holes_per_polygon_range
        self.points_per_hole_range = points_per_hole_range

        self.random = random.Random(random_seed)

    @staticmethod
    def create_box() -> shapely.geometry.Polygon:
        return shapely.geometry.Polygon(((0, 0), (1, 0), (1, 1), (0, 1)))

    @staticmethod
    def normalize_geometry(geometry: shapely.geometry.Polygon):
        bounds = geometry.bounds
        geometry = shapely.affinity.translate(geometry, (0 - bounds[0]), (0 - bounds[1]))

        scale = 1.0 / max(bounds[2] - bounds[0], bounds[3] - bounds[1])
        geometry = shapely.affinity.scale(geometry, scale, scale, origin=(0, 0, 0))

        return geometry
    
    @staticmethod
    def plot_geometry(geometry: shapely.geometry.Polygon):
        fig = plt.figure(1, figsize=(5,5), dpi=90)
        ax = fig.add_subplot(111)
        plot_polygon(geometry, ax, facecolor=(0.9, 0.9, 0.9), edgecolor=(0, 0, 0))
        plt.show()
    
    def _random_float(self) -> float:
        return float(self.random.randint(0, 1000)) / 1000
    
    def _random_coordinates(self, num_coordinates: int, bounds: Tuple[int, int, int, int] = None) -> List[Tuple[float, float]]:
        if bounds is None:
            bounds = (0, 0, 1, 1)
        return [(bounds[0] + self._random_float() * (bounds[2] - bounds[0]), bounds[1] + self._random_float() * (bounds[3] - bounds[1])) for i in range(num_coordinates)]
    
    # def random_coordinates_in_polygon(self, polygon : shapely.geometry.Polygon, num_coordinates):
    #     coordinates = []
    #     while len(coordinates) < num_coordinates:
    #         sampled_coordinate = polygon.point_on_surface()
    #         if sampled_coordinate not in coordinates:
    #             coordinates.append(sampled_coordinate)

    def _random_polygon(self, exterior_polygon: shapely.geometry.Polygon = None, generate_holes: bool = True) -> shapely.geometry.Polygon:
        if exterior_polygon is None:
            num_points = self.random.randint(*self.points_per_polygon_range)
            
            bounds_for_diversity = [[0.5, 0, 1, 1], [0, 0, 0.5, 1], [0, 0.5, 1, 1], [0, 0, 1, 0.5]]
            self.random.shuffle(bounds_for_diversity)
            
            outer = self._random_coordinates(num_points//3, bounds_for_diversity[0]) + self._random_coordinates(num_points//3, bounds_for_diversity[1]) + self._random_coordinates(num_points - 2*num_points//3, bounds_for_diversity[2])
            
            exterior_polygon: shapely.geometry.Polygon = shapely.geometry.MultiPoint(outer).convex_hull
        
        if not generate_holes:
            return exterior_polygon
        
        holes = []
        for _ in range(self.random.randint(*self.holes_per_polygon_range)):
            num_points = self.random.randint(*self.points_per_hole_range)
            while True:
                sampled_interior_coordinates = self._random_coordinates(num_points, exterior_polygon.bounds)
                # sampled_interior_coordinates = self.random_coordinates_in_polygon(exterior_polygon, num_points)
                
                interior_polygon: shapely.geometry.Polygon = shapely.geometry.Polygon(sampled_interior_coordinates).convex_hull

                if exterior_polygon.contains_properly(interior_polygon) and len([1 for hole in holes if interior_polygon.intersects(shapely.geometry.LinearRing(hole))]) == 0:
                    holes.append(interior_polygon.exterior.coords[::-1])
                    break
        return shapely.geometry.Polygon(exterior_polygon.exterior.coords, holes)

    def _random_multi_polygon(self) -> shapely.geometry.MultiPolygon:
        num_polygons = self.random.randint(*self.num_polygons_range)
        return shapely.geometry.MultiPolygon([self._random_polygon(generate_holes=False) for _ in range(num_polygons)])

    def generate_geometry(self) -> shapely.geometry.Polygon:
        geometry = self._random_multi_polygon()
        geometry : shapely.geometry.Polygon = unary_union(geometry)
        geometry = self._random_polygon(exterior_polygon=geometry)
        return geometry
    
    @staticmethod
    def _get_geometry_coordinates(geometry: shapely.geometry.Polygon) -> Tuple[List[List[Tuple[float, float]]], List[Tuple[float, float]]]:
        internal_polygons_coordinates = []
        for polygon in geometry.interiors:
            internal_polygons_coordinates.append(polygon.coords[:-1]) # The last point is the same as the first point
        
        external_polygon_coordinates = geometry.exterior.coords[:-1] # The last point is the same as the first point

        return internal_polygons_coordinates, external_polygon_coordinates

    @staticmethod
    def _create_gmsh_polygon(coordinates: List[Tuple[float, float]], mesh_size: float) -> Tuple[List[int], OrderedDict[int, Tuple[int, int]], int]:
        if(gmsh.isInitialized() == 0):
            raise Exception("Gmsh is not initialized")

        polygon_ptags = [] # List of tags of points that make up the polygon
        polygon_ltag_ptags = OrderedDict() # Maps the line tag to the tags of points that make up the line

        for point in coordinates:
            polygon_ptags.append(gmsh.model.geo.add_point(point[0], point[1], 0.0, mesh_size))
        
        for i in range(len(polygon_ptags) - 1):
            polygon_ltag_ptags[gmsh.model.geo.add_line(polygon_ptags[i], polygon_ptags[i+1])] = (polygon_ptags[i], polygon_ptags[i+1])
        polygon_ltag_ptags[gmsh.model.geo.add_line(polygon_ptags[-1], polygon_ptags[0])] = (polygon_ptags[-1], polygon_ptags[0]) # Close the polygon
        
        polygon_tag = gmsh.model.geo.add_curve_loop(list(polygon_ltag_ptags.keys()))

        return polygon_ptags, polygon_ltag_ptags, polygon_tag

    def generate_mesh(self, geometry: shapely.geometry.Polygon, name: str, mesh_size: float = 1e-2, view_mesh: bool = False, filetype: str = 'mesh') -> Tuple[List[List[int]], List[OrderedDict[int, Tuple[int, int]]]]:
        gmsh.initialize() # Initialize the Gmsh API

        # Get the coordinates of the external and internal polygons
        internal_polygons_coordinates, external_polygon_coordinates = self._get_geometry_coordinates(geometry)

        # Create the Gmsh geometry

        internal_polygons_ptags: List[List[int]] = []
        internal_polygons_ltag_ptags: List[OrderedDict[int, Tuple[int, int]]] = []
        internal_polygons_tag: List[int] = []
        
        # Create the internal polygons
        for coordinates in internal_polygons_coordinates:
            internal_polygon_ptags, internal_polygon_ltag_ptags, internal_polygon_tag = self._create_gmsh_polygon(coordinates, mesh_size)

            internal_polygons_ptags.append(internal_polygon_ptags)
            internal_polygons_ltag_ptags.append(internal_polygon_ltag_ptags)
            internal_polygons_tag.append(internal_polygon_tag)
        
        # Create the external polygon
        external_polygon_ptags, external_polygon_ltag_ptags, external_polygon_tag = self._create_gmsh_polygon(external_polygon_coordinates, mesh_size)

        # Combine the tags of the external and internal polygons
        polygons_ptags = [external_polygon_ptags, *internal_polygons_ptags]
        polygons_ltag_ptags = [external_polygon_ltag_ptags, *internal_polygons_ltag_ptags]
                
        surface_tag = gmsh.model.geo.add_plane_surface([external_polygon_tag, *internal_polygons_tag]) # Create the surface

        gmsh.model.addPhysicalGroup(2, [surface_tag], name="surface") # Add the surface to a physical group
        
        gmsh.model.geo.synchronize() # Synchronize the Gmsh model
        
        gmsh.model.mesh.generate() # Generate the mesh

        gmsh.write("{}.geo_unrolled".format(name)) # Write the geometry to a file
        
        gmsh.write("{}.{}".format(name, filetype)) # Write the mesh to a file
        
        if view_mesh:
            if 'close' not in sys.argv:
                gmsh.fltk.run()
        # Close the Gmsh API
        gmsh.finalize()

        return polygons_ptags, polygons_ltag_ptags
    
    def sample_conditions(self, polygons_ptags: List[List[int]], polygons_ltag_ptags: List[OrderedDict[int, Tuple[int, int]]], num_conditions: int = 4) -> List[Dict]:
        conditions = []
        
        combined_polygons_ptags = [ptag for ptags in polygons_ptags for ptag in ptags]
        combined_edges_ptags = [ptags for polygon_ltag_ptags in polygons_ltag_ptags for ptags in polygon_ltag_ptags.values()]
        
        while(len(conditions) < num_conditions):
            i_combined_polygons_ptags = deepcopy(combined_polygons_ptags)
            i_combined_edges_ptags = deepcopy(combined_edges_ptags)
            # 1. Choose a random number of constraints between 1 and the number of edges/vertices - 1, N (since num of edges == num of vertices)
            # 2. Sample N number of edges without replacement

            sampled_edges = self.random.sample(i_combined_edges_ptags, self.random.randint(1, len(i_combined_edges_ptags)-1))

            # 3. Create a list of all vertices that are on the sampled edges
            vertices_on_sampled_edges = set()
            for edge in sampled_edges:
                vertices_on_sampled_edges.add(edge[0])
                vertices_on_sampled_edges.add(edge[1])

            # 4. Sample a random number of edges from the list of sampled edges to actually constrain
            edges_to_constrain = self.random.sample(sampled_edges, self.random.randint(1, len(sampled_edges)))

            # 5. Delete the vertices that are on the edges that are sampled to constrain from the list of vertices constituting sampled edges
            # 6. Choose the remaining vertices constituting sampled edges as constraints
            vertices_to_constrain = deepcopy(vertices_on_sampled_edges)
            for edge in edges_to_constrain:
                if edge[0] in vertices_to_constrain:
                    vertices_to_constrain.remove(edge[0])
                if edge[1] in vertices_to_constrain:
                    vertices_to_constrain.remove(edge[1])

            # 7. Delete the sampled edges to constrain from the original list of edges, and the vertices constituting sampled edges from the original list of vertices
            for edge in edges_to_constrain:
                i_combined_edges_ptags.remove(edge)

            for vertex in vertices_on_sampled_edges:
                i_combined_polygons_ptags.remove(vertex)

            
            # 8. Sample a random number of vertices from the list of vertices as forces
            try:
                point_forces = self.random.sample(i_combined_polygons_ptags, self.random.randint(1, len(i_combined_polygons_ptags)))
            except:
                point_forces = []
            
            # 9. Sample a random number of edges from the list of edges as forces
            edge_forces = self.random.sample(i_combined_edges_ptags, self.random.randint(0 if len(point_forces) >= 1 else 1, len(i_combined_edges_ptags)))

            condition = {
                'point_constraints': list(vertices_to_constrain),
                'edge_constraints': list(edges_to_constrain),
                'point_forces': list(point_forces),
                'edge_forces': list(edge_forces)
            }

            if condition not in conditions:
                conditions.append(condition)

        # Sample magnitudes for forces between 500N and 5000N
        sign = [-1, 1]
        for condition in conditions:
            condition['point_forces'] = [(point_force, (self.random.randint(500, 5000)*self.random.choice(sign), self.random.randint(500, 5000)*self.random.choice(sign))) for point_force in condition['point_forces']]
            condition['edge_forces'] = [(edge_force, (self.random.randint(500, 5000)*self.random.choice(sign), self.random.randint(500, 5000)*self.random.choice(sign))) for edge_force in condition['edge_forces']]

        return conditions