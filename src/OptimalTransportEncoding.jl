module OptimalTransportEncoding

using CUDA
using CairoMakie
using GeometryBasics
using GeometryBasics: Mesh
using LinearAlgebra
using Serialization
using Statistics
using cuPDLPx: LibcuPDLPx as Lib

CUDA.allowscalar(false)

include("optimal_transport.jl")

include("core.jl")

include("utils.jl")

end # module OptimalTransportEncoding
