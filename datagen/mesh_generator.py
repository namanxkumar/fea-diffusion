import random
import sys
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, OrderedDict, Tuple

import gmsh
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import shapely.affinity
import shapely.geometry
from shapely.ops import unary_union
from shapely.plotting import plot_polygon
from sklearn.cluster import AgglomerativeClustering, KMeans

MATERIALS = [
    {"name": "Steel", "youngs_modulus": 210000, "poissons_ratio": 0.3},
    {"name": "Aluminum", "youngs_modulus": 68900, "poissons_ratio": 0.33},
    {"name": "Copper", "youngs_modulus": 117000, "poissons_ratio": 0.34},
    {"name": "Brass", "youngs_modulus": 97000, "poissons_ratio": 0.33},
    {"name": "Titanium", "youngs_modulus": 105000, "poissons_ratio": 0.34},
    {"name": "Stainless Steel", "youngs_modulus": 195000, "poissons_ratio": 0.3},
    {"name": "Nickel", "youngs_modulus": 207000, "poissons_ratio": 0.31},
    {"name": "Zinc", "youngs_modulus": 100000, "poissons_ratio": 0.25},
    {"name": "Lead", "youngs_modulus": 17500, "poissons_ratio": 0.44},
    {"name": "Magnesium", "youngs_modulus": 46500, "poissons_ratio": 0.35},
    {"name": "Concrete", "youngs_modulus": 30000, "poissons_ratio": 0.2},
    {"name": "Wood", "youngs_modulus": 11000, "poissons_ratio": 0.35},
    {"name": "Glass", "youngs_modulus": 64000, "poissons_ratio": 0.22},
    {"name": "Plastic", "youngs_modulus": 3000, "poissons_ratio": 0.4},
    {"name": "Rubber", "youngs_modulus": 0.01, "poissons_ratio": 0.5},
    {"name": "Bronze", "youngs_modulus": 120000, "poissons_ratio": 0.34},
    {"name": "Tungsten", "youngs_modulus": 411000, "poissons_ratio": 0.28},
    {"name": "Silver", "youngs_modulus": 83000, "poissons_ratio": 0.37},
    {"name": "Gold", "youngs_modulus": 78000, "poissons_ratio": 0.44},
    {"name": "Platinum", "youngs_modulus": 168000, "poissons_ratio": 0.38},
]


class MeshGenerator:
    def __init__(
        self,
        num_polygons_range: Tuple[int, int] = (1, 3),
        points_per_polygon_range: Tuple[int, int] = (3, 8),
        holes_per_polygon_range: Tuple[int, int] = (0, 3),
        points_per_hole_range: Tuple[int, int] = (3, 4),
        num_regions: Tuple[int, int] = (1, 5),
        random_seed=None,
    ):
        self.num_polygons_range = num_polygons_range
        self.points_per_polygon_range = points_per_polygon_range
        self.holes_per_polygon_range = holes_per_polygon_range
        self.points_per_hole_range = points_per_hole_range
        self.num_regions = num_regions
        self.random = random.Random(random_seed)

        self.mesh_path: str = None

    @staticmethod
    def create_box() -> shapely.geometry.Polygon:
        return shapely.geometry.Polygon(((0, 0), (1, 0), (1, 1), (0, 1)))

    @staticmethod
    def normalize_geometry(geometry: shapely.geometry.Polygon):
        bounds = geometry.bounds
        geometry = shapely.affinity.translate(
            geometry, (0 - bounds[0]), (0 - bounds[1])
        )

        scale = 1.0 / max(bounds[2] - bounds[0], bounds[3] - bounds[1])
        geometry = shapely.affinity.scale(geometry, scale, scale, origin=(0, 0, 0))

        return geometry

    @staticmethod
    def plot_geometry(geometry: shapely.geometry.Polygon):
        fig = plt.figure(1, figsize=(5, 5), dpi=90)
        ax = fig.add_subplot(111)
        plot_polygon(geometry, ax, facecolor=(0.9, 0.9, 0.9), edgecolor=(0, 0, 0))
        plt.show()

    def _random_float(self) -> float:
        return float(self.random.randint(0, 1000)) / 1000

    def _random_coordinates(
        self, num_coordinates: int, bounds: Tuple[int, int, int, int] = None
    ) -> List[Tuple[float, float]]:
        if bounds is None:
            bounds = (0, 0, 1, 1)
        return [
            (
                bounds[0] + self._random_float() * (bounds[2] - bounds[0]),
                bounds[1] + self._random_float() * (bounds[3] - bounds[1]),
            )
            for _ in range(num_coordinates)
        ]

    # def random_coordinates_in_polygon(self, polygon : shapely.geometry.Polygon, num_coordinates):
    #     coordinates = []
    #     while len(coordinates) < num_coordinates:
    #         sampled_coordinate = polygon.point_on_surface()
    #         if sampled_coordinate not in coordinates:
    #             coordinates.append(sampled_coordinate)

    def _random_polygon(
        self,
        exterior_polygon: shapely.geometry.Polygon = None,
        generate_holes: bool = True,
    ) -> shapely.geometry.Polygon:
        if exterior_polygon is None:
            num_points = self.random.randint(*self.points_per_polygon_range)

            bounds_for_diversity = [
                [0.5, 0, 1, 1],
                [0, 0, 0.5, 1],
                [0, 0.5, 1, 1],
                [0, 0, 1, 0.5],
            ]
            self.random.shuffle(bounds_for_diversity)

            outer = (
                self._random_coordinates(num_points // 3, bounds_for_diversity[0])
                + self._random_coordinates(num_points // 3, bounds_for_diversity[1])
                + self._random_coordinates(
                    num_points - 2 * num_points // 3, bounds_for_diversity[2]
                )
            )

            exterior_polygon: shapely.geometry.Polygon = shapely.geometry.MultiPoint(
                outer
            ).convex_hull

        if not generate_holes:
            return exterior_polygon

        holes = []
        for _ in range(self.random.randint(*self.holes_per_polygon_range)):
            num_points = self.random.randint(*self.points_per_hole_range)
            while True:
                sampled_interior_coordinates = self._random_coordinates(
                    num_points, exterior_polygon.bounds
                )
                # sampled_interior_coordinates = self.random_coordinates_in_polygon(exterior_polygon, num_points)

                interior_polygon: shapely.geometry.Polygon = shapely.geometry.Polygon(
                    sampled_interior_coordinates
                ).convex_hull

                if (
                    exterior_polygon.contains_properly(interior_polygon)
                    and len(
                        [
                            1
                            for hole in holes
                            if interior_polygon.intersects(
                                shapely.geometry.LinearRing(hole)
                            )
                        ]
                    )
                    == 0
                ):
                    holes.append(interior_polygon.exterior.coords[::-1])
                    break
        return shapely.geometry.Polygon(exterior_polygon.exterior.coords, holes)

    def _random_multi_polygon(self) -> shapely.geometry.MultiPolygon:
        num_polygons = self.random.randint(*self.num_polygons_range)
        return shapely.geometry.MultiPolygon(
            [self._random_polygon(generate_holes=False) for _ in range(num_polygons)]
        )

    def generate_geometry(self) -> shapely.geometry.Polygon:
        geometry = self._random_multi_polygon()
        geometry: shapely.geometry.Polygon = unary_union(geometry)
        geometry = self._random_polygon(exterior_polygon=geometry)
        return geometry

    @staticmethod
    def _get_geometry_coordinates(
        geometry: shapely.geometry.Polygon,
    ) -> Tuple[List[List[Tuple[float, float]]], List[Tuple[float, float]]]:
        internal_polygons_coordinates = []
        for polygon in geometry.interiors:
            internal_polygons_coordinates.append(
                polygon.coords[:-1]
            )  # The last point is the same as the first point

        external_polygon_coordinates = geometry.exterior.coords[
            :-1
        ]  # The last point is the same as the first point

        return internal_polygons_coordinates, external_polygon_coordinates

    @staticmethod
    def _create_gmsh_polygon(
        coordinates: List[Tuple[float, float]], mesh_size: float
    ) -> Tuple[List[int], OrderedDict[int, Tuple[int, int]], int]:
        if gmsh.isInitialized() == 0:
            raise Exception("Gmsh is not initialized")

        polygon_ptags = []  # List of tags of points that make up the polygon
        polygon_ltag_ptags = (
            OrderedDict()
        )  # Maps the line tag to the tags of points that make up the line

        for point in coordinates:
            polygon_ptags.append(
                gmsh.model.geo.add_point(point[0], point[1], 0.0, mesh_size)
            )

        for i in range(len(polygon_ptags) - 1):
            polygon_ltag_ptags[
                gmsh.model.geo.add_line(polygon_ptags[i], polygon_ptags[i + 1])
            ] = (polygon_ptags[i], polygon_ptags[i + 1])
        polygon_ltag_ptags[
            gmsh.model.geo.add_line(polygon_ptags[-1], polygon_ptags[0])
        ] = (
            polygon_ptags[-1],
            polygon_ptags[0],
        )  # Close the polygon

        polygon_tag = gmsh.model.geo.add_curve_loop(list(polygon_ltag_ptags.keys()))

        return polygon_ptags, polygon_ltag_ptags, polygon_tag

    def generate_mesh(
        self,
        geometry: shapely.geometry.Polygon,
        name: str,
        mesh_size: float = 1e-2,
        view_mesh: bool = False,
        filetype: str = "mesh",
    ) -> Tuple[List[List[int]], List[OrderedDict[int, Tuple[int, int]]]]:
        gmsh.initialize()  # Initialize the Gmsh API

        # Get the coordinates of the external and internal polygons
        (
            internal_polygons_coordinates,
            external_polygon_coordinates,
        ) = self._get_geometry_coordinates(geometry)

        # Create the Gmsh geometry

        internal_polygons_ptags: List[List[int]] = []
        internal_polygons_ltag_ptags: List[OrderedDict[int, Tuple[int, int]]] = []
        internal_polygons_tag: List[int] = []

        # Create the internal polygons
        for coordinates in internal_polygons_coordinates:
            (
                internal_polygon_ptags,
                internal_polygon_ltag_ptags,
                internal_polygon_tag,
            ) = self._create_gmsh_polygon(coordinates, mesh_size)

            internal_polygons_ptags.append(internal_polygon_ptags)
            internal_polygons_ltag_ptags.append(internal_polygon_ltag_ptags)
            internal_polygons_tag.append(internal_polygon_tag)

        # Create the external polygon
        (
            external_polygon_ptags,
            external_polygon_ltag_ptags,
            external_polygon_tag,
        ) = self._create_gmsh_polygon(external_polygon_coordinates, mesh_size)

        # Combine the tags of the external and internal polygons
        polygons_ptags = [external_polygon_ptags, *internal_polygons_ptags]
        polygons_ltag_ptags = [
            external_polygon_ltag_ptags,
            *internal_polygons_ltag_ptags,
        ]

        surface_tag = gmsh.model.geo.add_plane_surface(
            [external_polygon_tag, *internal_polygons_tag]
        )  # Create the surface

        gmsh.model.addPhysicalGroup(
            2, [surface_tag], name="surface"
        )  # Add the surface to a physical group

        gmsh.model.geo.synchronize()  # Synchronize the Gmsh model

        gmsh.model.mesh.generate()  # Generate the mesh

        gmsh.write("{}.geo_unrolled".format(name))  # Write the geometry to a file

        self.mesh_path = "{}.{}".format(name, filetype)
        gmsh.write(self.mesh_path)  # Write the mesh to a file

        if view_mesh:
            if "close" not in sys.argv:
                gmsh.fltk.run()
        # Close the Gmsh API
        gmsh.finalize()

        return polygons_ptags, polygons_ltag_ptags

    def _create_regions_with_kmeans(self) -> List[List]:
        mesh = pv.read(self.mesh_path)
        coords = np.array(mesh.points)
        num_clusters = random.randint(5, 20)  # Randomly select number of clusters

        clustering = KMeans(n_clusters=num_clusters)
        cluster_labels = clustering.fit_predict(coords)

        # num_regions = 1
        num_regions = random.randint(*self.num_regions)

        clustering2_centres = KMeans(n_clusters=num_regions)
        cluster2_labels_centres = clustering2_centres.fit_predict(
            clustering.cluster_centers_.reshape(-1, 1)
        )

        new_labels = np.empty_like(cluster_labels)
        for i in range(num_clusters):
            points_in_cluster = cluster_labels == i
            new_labels[points_in_cluster] = cluster2_labels_centres[i]

        # Initialize list to store coordinates for each region
        region_coordinates = [[] for _ in range(num_regions)]

        for i in range(num_clusters):
            points_in_cluster = cluster_labels == i
            region_idx = int(
                new_labels[points_in_cluster][0]
            )  # Get the region index for the cluster
            coords_in_cluster = coords[points_in_cluster][:, :2]
            region_coordinates[region_idx].extend(
                zip(coords_in_cluster[:, 0], coords_in_cluster[:, 1])
            )
        return region_coordinates

    def _create_regions_with_agglomerative_clustering(
        self, link: str
    ) -> List[List[Tuple[float, float]]]:
        mesh = pv.read(self.mesh_path)
        coords = np.array(mesh.points)

        # Randomly select number of regions
        # num_regions = 1
        num_regions = random.randint(*self.num_regions)

        agg_clustering = AgglomerativeClustering(n_clusters=num_regions, linkage=link)
        region_assignments = agg_clustering.fit_predict(coords)

        # Initialize list to store coordinates for each region
        region_coordinates = [[] for _ in range(num_regions)]

        for i in range(num_regions):
            points_in_region = region_assignments == i
            coords_in_region = coords[points_in_region][:, :2]
            region_coordinates[i].extend(
                zip(coords_in_region[:, 0], coords_in_region[:, 1])
            )

        return region_coordinates

    def _create_regions_randomly(self) -> List[List]:
        method = random.choice(["kmeans", "agglomerative"])
        if method == "kmeans":
            return self._create_regions_with_kmeans()
        else:
            link = random.choice(["complete", "average", "ward"])
            return self._create_regions_with_agglomerative_clustering(link)

    @staticmethod
    def _assign_materials_to_regions(
        regions: List[List[Tuple[float, float]]],
    ) -> Dict[Tuple[float, float], List[Tuple[float, float]]]:
        materials_dict = {}

        # Randomly select material properties for each region
        for i in range(len(regions)):
            material = random.choice(MATERIALS)
            material_properties = (
                material["youngs_modulus"],
                material["poissons_ratio"],
            )
            materials_dict[material_properties] = regions[i]
        return materials_dict

    def sample_conditions(
        self,
        polygons_ptags: List[List[int]],
        polygons_ltag_ptags: List[OrderedDict[int, Tuple[int, int]]],
        num_conditions: int = 4,
    ) -> List[Dict]:
        conditions = []

        combined_polygons_ptags = [ptag for ptags in polygons_ptags for ptag in ptags]
        combined_edges_ptags = [
            ptags
            for polygon_ltag_ptags in polygons_ltag_ptags
            for ptags in polygon_ltag_ptags.values()
        ]

        while len(conditions) < num_conditions:
            i_combined_polygons_ptags = deepcopy(combined_polygons_ptags)
            i_combined_edges_ptags = deepcopy(combined_edges_ptags)
            # 1. Choose a random number of constraints between 1 and the number of edges/vertices - 1, N (since num of edges == num of vertices)
            # 2. Sample N number of edges without replacement

            sampled_edges = self.random.sample(
                i_combined_edges_ptags,
                self.random.randint(1, len(i_combined_edges_ptags) - 1),
            )

            # 3. Create a list of all vertices that are on the sampled edges
            vertices_on_sampled_edges = set()
            for edge in sampled_edges:
                vertices_on_sampled_edges.add(edge[0])
                vertices_on_sampled_edges.add(edge[1])

            # 4. Sample a random number of edges from the list of sampled edges to actually constrain
            edges_to_constrain = self.random.sample(
                sampled_edges, self.random.randint(1, len(sampled_edges))
            )

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
                point_forces = self.random.sample(
                    i_combined_polygons_ptags,
                    self.random.randint(1, len(i_combined_polygons_ptags)),
                )
            except:
                point_forces = []

            # 9. Sample a random number of edges from the list of edges as forces
            edge_forces = self.random.sample(
                i_combined_edges_ptags,
                self.random.randint(
                    0 if len(point_forces) >= 1 else 1, len(i_combined_edges_ptags)
                ),
            )

            region_coordinates = self._create_regions_randomly()
            materials_assigned = self._assign_materials_to_regions(region_coordinates)

            condition = {
                "material_regions": dict(materials_assigned),
                "point_constraints": list(vertices_to_constrain),
                "edge_constraints": list(edges_to_constrain),
                "point_forces": list(point_forces),
                "edge_forces": list(edge_forces),
            }

            if condition not in conditions:
                conditions.append(condition)

        # Sample magnitudes for forces between 500N and 5000N
        sign = [-1, 1]
        for condition in conditions:
            condition["point_forces"] = [
                (
                    point_force,
                    (
                        self.random.randint(500, 5000) * self.random.choice(sign),
                        self.random.randint(500, 5000) * self.random.choice(sign),
                    ),
                )
                for point_force in condition["point_forces"]
            ]
            condition["edge_forces"] = [
                (
                    edge_force,
                    (
                        self.random.randint(500, 5000) * self.random.choice(sign),
                        self.random.randint(500, 5000) * self.random.choice(sign),
                    ),
                )
                for edge_force in condition["edge_forces"]
            ]

        return conditions
