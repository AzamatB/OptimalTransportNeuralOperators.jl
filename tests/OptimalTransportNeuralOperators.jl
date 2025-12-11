using GeometryBasics
using FileIO
using MeshIO

file_path = "datasets/ShapeNet-Car/data/mesh_001.ply"
mesh = load(file_path)
point_cloud = extract_vertices(mesh)

n = 64
torus = Torus(n)
point_cloud_latent = LatentPointCloud{Matrix{Float32}}(torus)

(encoding_indices, decoding_indices, ot_plan, point_cloud_transported) = compute_encoder_and_decoder(
    point_cloud, point_cloud_latent
)

unique(encoding_indices)
unique(decoding_indices)

encoded_points = point_cloud.points[:, encoding_indices]
decoded_points = point_cloud_transported.points[:, decoding_indices]

count(iszero, ot_plan.plan)/length(ot_plan.plan)
minimum(sum(ot_plan.plan; dims=2)) / (1 / size(ot_plan.plan, 1))
minimum(sum(ot_plan.plan; dims=1)) / (1 / size(ot_plan.plan, 2))
