import warnings
import numpy as np
import gudhi
from gudhi.wasserstein import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Voronoi, KDTree
from tqdm import tqdm
import matplotlib.pyplot as plt

D = 3  # dimension of input data
N = 2  # dimension of latent manifold


def topological_similarity_measurement(points1, points2, complex_mode='rips', dist_model='W'):
    # we assume points2 is interpolated from points1
    assert points2.shape[0] >= points1.shape[0]

    # Step1: Applying Persistent Homology
    # The Vietoris-Rips complex is a simplicial complex built as the clique-complex of a proximity graph
    if complex_mode == 'rips':
        # create rips complex1
        rips1 = gudhi.RipsComplex(points=points1, max_edge_length=2)
        simplex_tree1 = rips1.create_simplex_tree(max_dimension=2)
        # compute persistence diagram1
        simplex_tree1.compute_persistence(homology_coeff_field=2, min_persistence=0)
        diag1 = np.vstack([simplex_tree1.persistence_intervals_in_dimension(0),  # edges
                           simplex_tree1.persistence_intervals_in_dimension(1),  # rings
                           simplex_tree1.persistence_intervals_in_dimension(2)])  # holes
        # create rips complex2
        rips2 = gudhi.RipsComplex(points=points2, max_edge_length=2)
        simplex_tree2 = rips2.create_simplex_tree(max_dimension=2)
        # compute persistence diagram2
        simplex_tree2.compute_persistence(homology_coeff_field=2, min_persistence=0)
        diag2 = np.vstack([simplex_tree2.persistence_intervals_in_dimension(0),
                           simplex_tree2.persistence_intervals_in_dimension(1),
                           simplex_tree2.persistence_intervals_in_dimension(2)])

    else:  # TODO: we can try more simplicial complexes
        raise NotImplementedError

    # for debug and visualization
    # print("diag=", diag1)
    # gudhi.plot_persistence_diagram(diag1)
    # plt.show()
    # print("diag=", diag2)
    # gudhi.plot_persistence_diagram(diag2)
    # plt.show()

    # Step2: Measuring the similarity between two persistence diagrams
    if dist_model == 'W':  # Wasserstein distance
        diff = wasserstein_distance(diag1, diag2, matching=False, order=1, internal_p=2)
    elif dist_model == 'B':  # Bottleneck distance
        diff = gudhi.bottleneck_distance(diag1, diag2)
    else:  # TODO: we can try more measurements
        raise NotImplementedError
    return diff


def interpolate_near_underlying_manifold(points_xyz, K_max=8, K_min=4, threshold=0.20):
    # initialize some values
    m = points_xyz.shape[0]
    I = set()
    interpolated_points = []

    # For large scope, neighbourhoods will display topological features
    nbrs_max = NearestNeighbors(n_neighbors=K_max+1).fit(points_xyz)
    _, indices_max = nbrs_max.kneighbors(points_xyz)
    # For small scope, manifolds can be approximated by tangent planes
    nbrs_min = NearestNeighbors(n_neighbors=K_min+1).fit(points_xyz)
    _, indices_min = nbrs_min.kneighbors(points_xyz)

    # main loop for each point in the point cloud
    for l in tqdm(range(m)):
        if l not in I:
            I.add(l)

            # Step1: we check large scope
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
            # Obtain the index of K_max nearest neighbor points of x_l (not excluding itself)
            neighbors_indices = indices_max[l, :K_max+1]
            neigh_points = points_xyz[neighbors_indices]

            # Calculate Voronoi diagram
            vor = Voronoi(neigh_points)

            # Find the vertices in the Voronoi region and add them as interpolated points
            vertices = vor.vertices
            neigh_points_new = np.vstack([neigh_points, vertices])

            # catch warnings related to wasserstein_distance
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Cardinality of essential parts differs.",
                    category=UserWarning,
                )
                # Compute topological similarity
                similarity = topological_similarity_measurement(neigh_points, neigh_points_new)

            if similarity < threshold:
                I.update(neighbors_indices)
                interpolated_points.extend(vertices)
                continue
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

            # Step2: then, we check small scope
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
            # Obtain the index of K_max nearest neighbor points of x_l (excluding itself)
            neighbors_indices = indices_min[l, 1:K_min+1]

            # Using PCA to project points onto N-dimensional space (2-dimension)
            neighbors = points_xyz[neighbors_indices]
            pca = PCA(n_components=N)
            projected_points = pca.fit_transform(np.vstack([points_xyz[l], neighbors]))

            # Calculate Voronoi diagram
            vor = Voronoi(projected_points)

            # Find the vertices in the Voronoi region
            vertices = vor.vertices

            # Using tangent space approximation to generate new interpolation points
            U = pca.components_[:N]
            x_l = points_xyz[l]
            new_points = vertices @ U + x_l

            # Add new interpolation points
            I.update(neighbors_indices)
            interpolated_points.extend(new_points)

    # Return the interpolated point set
    return np.array(interpolated_points)


def estimate_curvature(points, k):
    """
    估计点云的高斯曲率（这里使用局部PCA方法）
    """
    # 假设points是一个N x 3的数组，其中N是点的数量
    curvatures = np.zeros(len(points))
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    _, indices = nbrs.kneighbors(points)

    for i in tqdm(range(len(points))):
        neighbors = points[indices[i]]
        centroid = np.mean(neighbors, axis=0)
        cov_matrix = np.cov(neighbors - centroid, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        curvatures[i] = eigenvalues[0] / np.sum(eigenvalues)

    return curvatures, nbrs


def generate_midpoints(points, k, curvature_threshold):
    """
    生成中点集
    """
    curvatures, nbrs = estimate_curvature(points, k)
    low_curvature_points = points[curvatures < curvature_threshold]
    _, indices = nbrs.kneighbors(low_curvature_points)

    midpoints = []
    for i in tqdm(range(len(low_curvature_points))):
        p = low_curvature_points[i]
        for j in indices[i]:
            pj = points[j]
            midpoint = (p + pj) / 2
            midpoints.append(midpoint)

    return np.array(midpoints)


def interpolate_with_midpoints(points, K, threshold):
    midpoints = generate_midpoints(points, K, threshold)
    aug_points = np.vstack([points, midpoints])
    return aug_points


def generate_random_points(points, Np):  # Np: Number of additional points to generate
    min_bounds = np.min(points, axis=0)
    max_bounds = np.max(points, axis=0)
    random_points = np.random.uniform(min_bounds, max_bounds, size=(Np, 3))
    return random_points


def interpolate_with_random(points, Np):  # Np: Number of additional points to generate
    random_points = generate_random_points(points, Np)
    all_points = np.vstack([points, random_points])
    return all_points


def linear_interpolate(points, random_point):
    weights = 1 / np.linalg.norm(points - random_point, axis=1)
    weights /= np.sum(weights)
    interpolated_point = np.dot(weights, points)
    return interpolated_point


def interpolate_with_tree(points, K, Np, td):
    tree = KDTree(points)
    new_points = generate_random_points(points, Np)

    all_distances = []
    for p in tqdm(new_points):
        distances, knn_indices = tree.query(p, k=K)
        P_knn = points[knn_indices]
        valid_indices = np.where(distances > td)[0]
        P_valid = P_knn[valid_indices]

        all_distances.append(distances)

        if len(P_valid) > 0:
            p_c = linear_interpolate(P_valid, p)
            points = np.vstack([points, p_c])

    all_distances = np.asarray(all_distances).flatten()
    return points


if __name__ == "__main__":
    # Example data: 3D point cloud
    # points_xyz = np.fromfile('./pc.bin', dtype=np.float32).reshape(-1, 3)
    points_xyz = np.random.random((100, 3))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2], c='blue', label='Original Points')
    # plt.show()

    # Execute our interpolation algorithm
    interpolated_points = interpolate_near_underlying_manifold(points_xyz, K_max=15, K_min=7, threshold=0.25)

    # Visualization
    ax.scatter(interpolated_points[:, 0], interpolated_points[:, 1], interpolated_points[:, 2], c='red', label='Interpolated Points')
    ax.legend()
    plt.show()
