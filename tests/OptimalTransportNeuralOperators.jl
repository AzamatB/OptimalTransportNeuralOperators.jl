using GeometryBasics
using FileIO
using MeshIO

file_path = "datasets/ShapeNet-Car/data/mesh_001.ply"
mesh = load(file_path)
points = extract_vertices(mesh)

R = 3.0
r = 1.0
nθ = 64
nφ = 64

(latent_points, θ_vals, φ_vals) = generate_torus_latent_points(R, r, nθ, nφ)

dists = pairwise_squared_euclidean_distance(points, latent_points)

@time plan = sinkhorn_plan(points, latent_points)

count(iszero, plan)/length(plan)
minimum(sum(plan; dims=2))/(1/size(plan, 1))
minimum(sum(plan; dims=1))/(1/size(plan, 2))
