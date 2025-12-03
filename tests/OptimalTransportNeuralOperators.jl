using GeometryBasics
using FileIO
using MeshIO

file_path = "datasets/ShapeNet-Car/data/mesh_001.ply"
mesh = load(file_path)
points = extract_vertices(mesh)

R = 3.0
r = 1.0
nθ = 50
nφ = 50

(latent_points, θ_vals, φ_vals) = generate_torus_latent_points(R, r, nθ, nφ)

dists = pairwise_squared_euclidean_distance(points, latent_points)

sinkhorn_plan(points, latent_points)
