using NPZ

M = Matrix{Float32}
k = 11
file_path_x = "datasets/ShapeNet-Car/data/mesh_0$k.ply"
file_path_y = "datasets/ShapeNet-Car/data/press_0$k.npy"
mesh = load(file_path_x)
measure = OrientedSurfaceMeasure{M}(mesh)

n = ceil(Int, 1.7 * measure.num_points)
measure_l = LatentOrientedSurfaceMeasure{M}(Torus(n))

@time (features, encoding_indices, decoding_indices) = encode(measure, measure_l)

unique(encoding_indices)
unique(decoding_indices)

target = Float32.(npzread(file_path_y))
ŷ = features[:, :, 1, 1]
target - vec(ŷ)[decoding_indices]
