using GeometryBasics
using FileIO
using MeshIO
using NNlib
using Statistics

abstract type LatentGrid end

# As ϕ changes from 0 to 2π, a point traces out a full circle around the z-axis,
# forming the outer "ring" of the torus.
# As θ changes from 0 to 2π, a point moves around the cross-sectional circle of the tube.
struct Torus <: LatentGrid
    n::Int
    R::Float32
    r::Float32
    ϕ_vals::Vector{Float32}
    θ_vals::Vector{Float32}

    function Torus(n::Int, R::Real=2.0f0, r::Real=1.0f0)
        @assert R > r > 0
        T = Float32
        ϕ_vals = collect(range(T(0), T(2π), length=n + 1)[1:end-1])
        θ_vals = collect(range(T(0), T(2π), length=n + 1)[1:end-1])
        return new(n, R, r, ϕ_vals, θ_vals)
    end
end

abstract type AbstractPointCloud{M<:DenseMatrix{Float32}} end

struct EuclideanPointCloud{M<:DenseMatrix{Float32}} <: AbstractPointCloud{M}
    num_points::Int
    points::M
end

function EuclideanPointCloud(points::M) where {M<:DenseMatrix{Float32}}
    num_points = size(points, 2)
    return EuclideanPointCloud{M}(num_points, points)
end

struct LatentPointCloud{M<:DenseMatrix{Float32},G<:LatentGrid} <: AbstractPointCloud{M}
    num_points::Int
    points::M
    grid::G
end

function LatentPointCloud(points::M, grid::G) where {M<:DenseMatrix{Float32},G<:LatentGrid}
    num_points = size(points, 2)
    return LatentPointCloud{M,G}(num_points, points, grid)
end

function LatentPointCloud{M}(torus::Torus) where {M<:DenseMatrix{Float32}}
    T = Float32
    n = torus.n
    R = torus.R
    r = torus.r
    ϕ_vals = torus.ϕ_vals
    θ_vals = torus.θ_vals

    points = Array{T,3}(undef, 3, n, n)
    @inbounds for j in eachindex(θ_vals)
        θ = θ_vals[j]
        v = R + r * cos(θ)
        z = r * sin(θ)
        for i in eachindex(ϕ_vals)
            ϕ = ϕ_vals[i]
            x = v * cos(ϕ)
            y = v * sin(ϕ)
            points[1, i, j] = x
            points[2, i, j] = y
            points[3, i, j] = z
        end
    end
    # 3 × n² (column-major, flattens ϕ first, then θ)
    points_mat = reshape(points, 3, :)
    num_points = size(points_mat, 2)
    point_cloud = LatentPointCloud{M,Torus}(num_points, points_mat, torus)
    return point_cloud
end

# extract vertices from a mesh as a 3×n Matrix{T}
function extract_vertices(mesh::Mesh)
    T = Float32
    vertices = mesh.vertex_attributes.position
    num_points = length(vertices)
    dim = length(first(vertices))
    points = Matrix{T}(undef, dim, num_points)
    eachcol(points) .= vertices
    point_cloud = EuclideanPointCloud(points)
    return point_cloud
end

# Computes pairwise squared Euclidean distance between two point clouds.
# xs: (d, n) - Latent points
# ys: (d, m) - Physical grid points
# Returns: (n, m) distance matrix
function pairwise_squared_euclidean_distance(
    xs::M,                                      # d × n
    ys::M                                       # d × m
) where {M<:DenseMatrix{Float32}}
    # xᵀx, yᵀy
    xs_sq = sum(abs2, xs; dims=1)               # (1 × n)
    ys_sq = sum(abs2, ys; dims=1)               # (1 × m)
    xs_sqᵀ = xs_sq'                             # (n × 1)
    # xᵀy
    xys = xs' * ys                              # (n × m)
    # xᵀx + yᵀy - 2xᵀy,  ∀ x ∈ xs, y ∈ ys
    dists = @. xs_sqᵀ + ys_sq - 2 * xys
    # ensure non-negativity
    zer = zero(Float32)
    dists = max.(dists, zer)
    return dists                                # (n × m)
end

function pairwise_squared_euclidean_distance(
    xs::AbstractPointCloud{M}, ys::AbstractPointCloud{M}
) where {M<:DenseMatrix{Float32}}
    return pairwise_squared_euclidean_distance(xs.points, ys.points)
end

"""
    compute_optimal_transport_plan(xs::M, ys::M, num_iters::Int=50)

    Compute an entropic regularized optimal transport plan between discrete measures on
xs (d×n) and ys (d×m) using the Log-Domain Sinkhorn-Knopp algorithm, assuming uniform marginals.
    Returns P (n×m) such that P >= 0.
"""
function compute_optimal_transport_plan(cost_mat::DenseMatrix{Float32}, num_iters::Int)
    T = Float32
    ε = T(0.005) # relative (to the mean cost) entropy regularization parameter
    (n, m) = size(cost_mat)

    # target log-marginals (uniform measures)
    # kept as (n, 1) and (1, m) to ensure clear broadcasting against cost_mat
    log_mu = similar(cost_mat, T, n, 1)
    log_nu = similar(cost_mat, T, 1, m)
    fill!(log_mu, T(-log(n)))
    fill!(log_nu, T(-log(m)))

    # initialize dual variables (potentials)
    f = similar(cost_mat, T, n, 1)
    g = similar(cost_mat, T, 1, m)
    zer = zero(T)
    fill!(f, zer)
    fill!(g, zer)

    # effective entropy regularization: ε_eff = ε * mean(cost_mat)
    c = -1.0f0 / (ε * mean(cost_mat))
    # kernel in log domain: logK = -cost_mat / (ε * mean(cost_mat))
    logK = c .* cost_mat

    for _ in 1:num_iters
        # update f (row normalization)
        S_f = g .+ logK                  # (1 × m) .+ (n × m) -> (n × m)
        lse_f = logsumexp(S_f; dims=2)
        f = log_mu .- lse_f
        # update g (column normalization)
        S_g = f .+ logK                  # (n × 1) .+ (n × m) -> (n × m)
        lse_g = logsumexp(S_g; dims=1)
        g = log_nu .- lse_g
    end
    # transport plan: log_P = f + g + logK; -> (n, 1) .+ (1, m) .+ (n, m) -> (n, m)
    P = exp.(f .+ g .+ logK)
    return P
end

struct OptimalTransportPlan{M<:DenseMatrix{Float32},G<:LatentGrid}
    plan::M
    grid::G
end

function OptimalTransportPlan(
    xs::EuclideanPointCloud{M}, ys::LatentPointCloud{M,G}, num_iters::Int=100
) where {M<:DenseMatrix{Float32},G<:LatentGrid}
    dists = pairwise_squared_euclidean_distance(xs, ys)
    plan = compute_optimal_transport_plan(dists, num_iters)
    return OptimalTransportPlan(plan, ys.grid)
end

function normalize_columns(xs::AbstractMatrix{<:Real})
    col_sums = sum(xs; dims=1)
    xs_norm = xs ./ col_sums
    return xs_norm
end

function transport(
    points::DenseMatrix{Float32},             # (d × n)
    plan::AbstractMatrix{Float32}             # (n × m)
)
    @assert size(points, 2) == size(plan, 1)
    # normalize the transport plan columns for barycentric projection
    plan_norm = normalize_columns(plan)       # (n × m)
    # perform the barycentric projection
    points_transported = points * plan_norm   # (d × m)
    return points_transported
end

"""
    pushforward_to_physical(point_cloud::EuclideanPointCloud, ot_plan::OptimalTransportPlan)

point_cloud: (d × n) features on physical surface.
ot_plan:     (n × m) optimal transport plan from physical to latent spaces.
Returns point_cloud_transported: (d × m) features on latent grid.
"""
function pushforward_to_physical(
    point_cloud::EuclideanPointCloud{M},                       # (d × n)
    ot_plan::OptimalTransportPlan{M,G}                         # (n × m)
) where {M<:DenseMatrix{Float32},G<:LatentGrid}
    plan = ot_plan.plan
    points_transported = transport(point_cloud.points, plan)   # (d × m)
    point_cloud_transported = EuclideanPointCloud(points_transported)
    return point_cloud_transported
end

"""
    pullback_from_latent(point_cloud::LatentPointCloud, ot_plan::OptimalTransportPlan)

point_cloud: (d × m) features on latent grid.
ot_plan:     (n × m) optimal transport plan from physical to latent spaces.
Returns point_cloud_transported: (d × n) features on physical surface.
"""
function pullback_from_latent(
    point_cloud::LatentPointCloud{M},                           # (d × m)
    ot_plan::OptimalTransportPlan{M}                            # (n × m)
) where {M<:DenseMatrix{Float32}}
    plan = ot_plan.plan
    # transpose the transport plan to go from latent to physical
    points_transported = transport(point_cloud.points, plan')   # (d × n)
    point_cloud_transported = EuclideanPointCloud(points_transported)
    return point_cloud_transported
end

# for each point in the destination point cloud, find and assign the index of the closest
# point in the source point cloud
function assign_points(
    point_cloud_src::AbstractPointCloud,       # (d x n)
    point_cloud_dst::AbstractPointCloud        # (d x m)
)
    dists = pairwise_squared_euclidean_distance(point_cloud_src, point_cloud_dst) # (n x m)
    index_pairs = vec(argmin(dists; dims=1))   # (m)
    indices_best = getindex.(index_pairs, 1)   # (m)
    return indices_best
end

function compute_encoder_and_decoder(
    point_cloud::EuclideanPointCloud{M},      # (d × n)
    point_cloud_latent::LatentPointCloud{M}   # (d × m)
) where {M<:DenseMatrix{Float32}}
    ot_plan = OptimalTransportPlan(point_cloud, point_cloud_latent)           # (n × m)
    point_cloud_transported = pushforward_to_physical(point_cloud, ot_plan)   # (d × m)
    encoding_indices = assign_points(point_cloud, point_cloud_transported)    # (m)
    decoding_indices = assign_points(point_cloud_transported, point_cloud)    # (n)
    return (encoding_indices, decoding_indices, ot_plan, point_cloud_transported)
end
