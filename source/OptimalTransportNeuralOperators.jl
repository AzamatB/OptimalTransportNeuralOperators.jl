using GeometryBasics
using FileIO
using LinearAlgebra
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

    sincosϕs = NTuple{2,Float32}[(sin(ϕ), cos(ϕ)) for ϕ in ϕ_vals]
    points = Array{T}(undef, 3, n, n)
    @inbounds for j in eachindex(θ_vals)
        θ = θ_vals[j]
        (sinθ, cosθ) = sincos(θ)
        v = R + r * cosθ
        z = r * sinθ
        for i in eachindex(ϕ_vals)
            (sinϕ, cosϕ) = sincosϕs[i]
            x = v * cosϕ
            y = v * sinϕ
            points[1,i,j] = x
            points[2,i,j] = y
            points[3,i,j] = z
        end
    end

    # 3 × n² (column-major, flattens ϕ first, then θ)
    points_mat = reshape(points, 3, :)
    point_cloud = LatentPointCloud{M,Torus}(points_mat, torus)
    return point_cloud
end

# extract vertices from a mesh as a 3×n Matrix{T}
function extract_vertices(mesh::Mesh)
    T = Float32
    vertices = mesh.position
    num_points = length(vertices)
    dim = length(first(vertices))
    points = Matrix{T}(undef, dim, num_points)
    eachcol(points) .= vertices
    point_cloud = EuclideanPointCloud(points)
    return point_cloud
end

# compute vertex normals and weights for a 3D mesh composed of triangular faces, which are
# computed by averaging the normals of adjacent faces, weighted by the area of those faces.
function compute_mesh_vertex_normals(
    mesh::Mesh{3,Float32,FaceType,(:position,),Tuple{Vector{Point{3,Float32}}}}
) where {FaceType<:NgonFace{3}}
    T = Float32
    faces = mesh.faces
    vertices = mesh.position
    num_vertices = length(vertices)
    vertex_normals = zeros(T, 3, num_vertices)
    vertex_weights = zeros(T, num_vertices)

    @inbounds for n in eachindex(faces)
        face = faces[n]
        i₁ = face[1]
        i₂ = face[2]
        i₃ = face[3]

        vertex₁ = vertices[i₁]
        vertex₂ = vertices[i₂]
        vertex₃ = vertices[i₃]

        edge₁ = vertex₂ - vertex₁
        edge₂ = vertex₃ - vertex₁
        normal = edge₁ × edge₂
        (normal_x, normal_y, normal_z) = normal

        vertex_normals[1,i₁] += normal_x
        vertex_normals[2,i₁] += normal_y
        vertex_normals[3,i₁] += normal_z

        vertex_normals[1,i₂] += normal_x
        vertex_normals[2,i₂] += normal_y
        vertex_normals[3,i₂] += normal_z

        vertex_normals[1,i₃] += normal_x
        vertex_normals[2,i₃] += normal_y
        vertex_normals[3,i₃] += normal_z

        # ‖normal‖ = 2⋅(area of face)
        double_area = norm(normal)
        vertex_weights[i₁] += double_area
        vertex_weights[i₂] += double_area
        vertex_weights[i₃] += double_area
    end

    # normalize vertex weights to sum to 1
    Σweight⁻¹ = inv(sum(vertex_weights))
    @assert isfinite(Σweight⁻¹)
    vertex_weights .*= Σweight⁻¹

    # normalize each vertex normal to unit length
    @inbounds for normal in eachcol(vertex_normals)
        magnitude = norm(normal)
        # avoid division by zero
        magnitude += iszero(magnitude)
        # view mutates the underlying array
        normal ./= magnitude
    end
    return (vertex_normals, vertex_weights)
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
Returns point_cloud_transported: (d × n) features on physical surface, i. e. what the latent
grid looks like after being bent/warped to match the surface of the input physical object.
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
