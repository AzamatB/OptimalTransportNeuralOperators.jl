using GeometryBasics
using FileIO
using MeshIO

file_path = "datasets/ShapeNet-Car/data/mesh_001.ply"
mesh = load(file_path)
point_cloud = extract_vertices(mesh)

n = 64
torus = Torus(n)
point_cloud_latent = LatentPointCloud{Matrix{Float32}}(torus)

dists = pairwise_squared_euclidean_distance(point_cloud, point_cloud_latent)
@time ot_plan = OptimalTransportPlan(point_cloud, point_cloud_latent)

point_cloud_transported_latent = pushforward_to_latent(point_cloud, ot_plan)
point_cloud_transported = pullback_to_physical(point_cloud_transported_latent, ot_plan)

indices = assign_points(point_cloud, point_cloud_transported_latent)

unique(indices)

count(iszero, ot_plan.plan)/length(ot_plan.plan)
minimum(sum(ot_plan.plan; dims=2)) / (1 / size(ot_plan.plan, 1))
minimum(sum(ot_plan.plan; dims=1)) / (1 / size(ot_plan.plan, 2))
