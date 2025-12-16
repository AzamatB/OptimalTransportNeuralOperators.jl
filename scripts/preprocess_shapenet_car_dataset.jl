module_path = normpath(joinpath(@__DIR__, "..", "src"))
push!(LOAD_PATH, module_path)

import OptimalTransportNeuralOperators as OTNO

using FileIO
using LazyArrays
using MeshIO
using NPZ
using OptimalTransportNeuralOperators

struct ShapeNetCarDataSamplePaths <: OTNO.AbstractDataSamplePaths
    mesh::String
    target::String
end

function OTNO.read_mesh_and_target(sample_paths::ShapeNetCarDataSamplePaths)
    mesh = load(sample_paths.mesh)
    target_raw = npzread(sample_paths.target)
    num_points = length(mesh.position)
    num_points_raw = length(target_raw)

    # remove elements 17:112 that do not have corresponding vertices in the mesh object
    indices = Vcat(1:16, 113:num_points_raw)
    target = map(i -> Float32(target_raw[i]), indices)
    @assert length(target) == num_points
    return (mesh, target)
end

function preprocess_and_save_dataset(
    dir_src::String, dir_dst::String, ::Type{M}
) where {M<:DenseMatrix{Float32}}
    folders = readdir(dir_src)
    for folder in folders
        subdir = joinpath(dir_src, folder)
        isdir(subdir) || continue

        file_path_mesh = joinpath(subdir, "tri_mesh.ply")
        file_path_target = joinpath(subdir, "press.npy")
        file_path_dst = joinpath(dir_dst, "$folder.jld")

        sample_paths = ShapeNetCarDataSamplePaths(file_path_mesh, file_path_target)
        data_sample = OTNO.OTNODataSample(sample_paths, M)
        OTNO.save_sample(file_path_dst, data_sample)
    end
    return nothing
end

#######   preprocess the ShapeNet-Car dataset and save processed files   #######

M = Matrix{Float32}
dir_src = "datasets/car-pressure-data/data"
dir_dst = "datasets/car-pressure-data/processed-data"

@time preprocess_and_save_dataset(dir_src, dir_dst, M)
