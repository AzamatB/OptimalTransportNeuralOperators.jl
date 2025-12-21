module OptimalTransportEncoding

using CUDA
using CairoMakie
using GeometryBasics
using GeometryBasics: Mesh
using LinearAlgebra
using NNlib
using Serialization
using Statistics
using cuPDLPx: LibcuPDLPx as Lib

CUDA.allowscalar(false)

include("core.jl")

include("optimal_transport.jl")

include("utils.jl")

end # module OptimalTransportEncoding
