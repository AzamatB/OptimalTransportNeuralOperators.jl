using GeometryBasics
using FileIO
using MeshIO

M = Matrix{Float32}
file_path = "datasets/ShapeNet-Car/data/mesh_001.ply"
mesh = load(file_path)
measure = OrientedSurfaceMeasure{M}(mesh)

n = ceil(Int, 1.2 * measure.num_points)
measure_l = LatentOrientedSurfaceMeasure{M}(Torus(n))

@time (measure_t, encoding_indices, decoding_indices, ot_plan) = encode(measure, measure_l)

unique(encoding_indices)
unique(decoding_indices)

estimate_plan_convergence(ot_plan, measure, measure_l)
