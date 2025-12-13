using CUDA
using GeometryBasics
using FileIO
using LinearAlgebra
using MeshIO
using NNlib
using Statistics

abstract type LatentGrid end

function vec_type(::Type{<:Array{T}}) where {T}
    return Vector{T}
end
function vec_type(::Type{<:CuArray{T,N,M}}) where {T,N,M}
    return CuVector{T,M}
end

# As ϕ changes from 0 to 2π, a point traces out a full circle around the z-axis,
# forming the outer "ring" of the torus.
# As θ changes from 0 to 2π, a point moves around the cross-sectional circle of the tube.
struct Torus <: LatentGrid
    n::Int
    R::Float32
    r::Float32
    ϕ_vals::Vector{Float32}
    θ_vals::Vector{Float32}

    function Torus(nθ::Int, R::Real=2.0f0, r::Real=1.0f0)
        @assert R > r > 0
        T = Float32
        nϕ = floor(Int, nθ * R / r)
        n = nϕ * nθ
        ϕ_vals = collect(range(T(0), T(2π), length=(nϕ + 1))[1:end-1])
        θ_vals = collect(range(T(0), T(2π), length=(nθ + 1))[1:end-1])
        return new(n, R, r, ϕ_vals, θ_vals)
    end
end

abstract type AbstractMeasure{M<:DenseMatrix{Float32}} end

struct OrientedSurfaceMeasure{M<:DenseMatrix{Float32},V<:DenseVector{Float32}} <: AbstractMeasure{M}
    num_points::Int
    points::M
    normals::Matrix{Float32}
    weights::V
end

# extract vertices as a point cloud from a 3D trianguler mesh object and compute
# corresponding vertex normals and weights by averaging the normals of adjacent faces,
# weighted by the area of those faces.
function OrientedSurfaceMeasure{M}(
    mesh::Mesh{3,Float32,FaceType,(:position,),Tuple{Vector{Point{3,Float32}}}}
) where {M,FaceType<:NgonFace{3}}
    T = Float32
    faces = mesh.faces
    vertices = mesh.position
    num_points = length(vertices)
    dim = 3

    points = Matrix{T}(undef, dim, num_points)
    eachcol(points) .= vertices
    normals = zeros(T, dim, num_points)
    weights = zeros(T, num_points)

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

        normals[1, i₁] += normal_x
        normals[2, i₁] += normal_y
        normals[3, i₁] += normal_z

        normals[1, i₂] += normal_x
        normals[2, i₂] += normal_y
        normals[3, i₂] += normal_z

        normals[1, i₃] += normal_x
        normals[2, i₃] += normal_y
        normals[3, i₃] += normal_z

        # ‖normal‖ = 2⋅(area of face)
        double_area = norm(normal)
        weights[i₁] += double_area
        weights[i₂] += double_area
        weights[i₃] += double_area
    end

    # normalize vertex weights to sum to 1
    Σweight⁻¹ = inv(sum(weights))
    @assert isfinite(Σweight⁻¹)
    weights .*= Σweight⁻¹

    VectorType = vec_type(M)
    weights_gpu = VectorType(weights)
    points_gpu = M(points)

    # normalize each vertex normal to unit length
    @inbounds for normal in eachcol(normals)
        magnitude = norm(normal)
        # avoid division by zero
        magnitude += iszero(magnitude)
        # view mutates the underlying array
        normal ./= magnitude
    end
    return OrientedSurfaceMeasure(num_points, points_gpu, normals, weights_gpu)
end

struct LatentOrientedSurfaceMeasure{M<:DenseMatrix{Float32},G<:LatentGrid,V<:DenseVector{Float32}} <: AbstractMeasure{M}
    num_points::Int
    points::M
    normals::Matrix{Float32}
    weights::V
    grid::G
end

function LatentOrientedSurfaceMeasure{M}(torus::Torus) where {M<:DenseMatrix{Float32}}
    T = Float32
    R = torus.R
    r = torus.r
    ϕ_vals = torus.ϕ_vals
    θ_vals = torus.θ_vals
    nϕ = length(ϕ_vals)
    nθ = length(θ_vals)
    num_points = nϕ * nθ
    dim = 3

    sincosϕs = NTuple{2,Float32}[(sin(ϕ), cos(ϕ)) for ϕ in ϕ_vals]
    points = Array{T}(undef, dim, nϕ, nθ)
    normals = Array{T}(undef, dim, nϕ, nθ)
    weights = Matrix{T}(undef, nϕ, nθ)

    @inbounds for j in eachindex(θ_vals)
        θ = θ_vals[j]
        (sinθ, cosθ) = sincos(θ)
        v = R + r * cosθ
        z = r * sinθ
        for i in eachindex(ϕ_vals)
            (sinϕ, cosϕ) = sincosϕs[i]
            # point (x, y, z)
            points[1,i,j] = v * cosϕ
            points[2,i,j] = v * sinϕ
            points[3,i,j] = z
            # unit normal vector at (x, y, z)
            normals[1,i,j] = cosϕ * cosθ
            normals[2,i,j] = sinϕ * cosθ
            normals[3,i,j] = sinθ
        end
        # weight ∝ (R + r * cosθ)
        weights[:,j] .= v
    end

    # flatten into (dim × num_points); column-major – flattens ϕ first, then θ
    points_mat = M(reshape(points, dim, :))
    normals_mat = reshape(normals, dim, :)
    weights_vec = reshape(weights, :)

    # normalize weights to sum to 1; normals are already unit length
    Σweight⁻¹ = inv(sum(weights_vec))
    @assert isfinite(Σweight⁻¹)
    weights_vec .*= Σweight⁻¹
    VectorType = vec_type(M)
    weights_gpu = VectorType(weights_vec)
    return LatentOrientedSurfaceMeasure(num_points, points_mat, normals_mat, weights_gpu, torus)
end

struct OptimalTransportPlan{M<:DenseMatrix{Float32},G<:LatentGrid}
    plan::M
    grid::G
end

function OptimalTransportPlan(
    xs::OrientedSurfaceMeasure{M}, ys::LatentOrientedSurfaceMeasure{M,G}, num_iters::Int=100
) where {M<:DenseMatrix{Float32},G<:LatentGrid}
    dists = pairwise_squared_euclidean_distance(xs.points, ys.points)
    plan = compute_optimal_transport_plan(dists, xs.weights, ys.weights, num_iters)
    return OptimalTransportPlan(plan, ys.grid)
end

"""
    compute_optimal_transport_plan(cost_mat, mu, nu, num_iters)

    Compute an entropic regularized optimal transport plan between discrete measures
    mu (length n) and nu (length m) given a cost matrix (n×m) using the
    Log-Domain Sinkhorn-Knopp algorithm.

    Arguments:
    - cost_mat: DenseMatrix{Float32} of size (n, m)
    - mu: Source marginals (vector of length n)
    - nu: Target marginals (vector of length m)
    - num_iters: Number of iterations

    Returns P (n×m) such that P >= 0, P*1 ≈ mu, P'*1 ≈ nu.
"""
function compute_optimal_transport_plan(
    cost_mat::DenseMatrix{Float32}, mu::V, nu::V, num_iters::Int
) where {V<:DenseVector{Float32}}
    T = Float32
    ε = T(0.005) # relative (to the mean cost) entropy regularization parameter
    (n, m) = size(cost_mat)

    # validate dimensions
    @assert length(mu) == n "Source marginal dimension (mu) does not match cost_mat rows"
    @assert length(nu) == m "Target marginal dimension (nu) does not match cost_mat columns"

    # target log-marginals: take log, and reshape for broadcasting
    log_mu = reshape(log.(mu), n, 1)   # (n × 1) to broadcast against columns
    log_nu = reshape(log.(nu), 1, m)   # (1 × m) to broadcast against rows

    # initialize dual variables (potentials)
    f = zero(log_mu)   # (n × 1)
    g = zero(log_nu)   # (1 × m)

    # effective entropy regularization: ε_eff = ε * mean(cost_mat)
    c = -1.0f0 / (ε * mean(cost_mat))
    # kernel in log domain: logK = -cost_mat / (ε * mean(cost_mat))
    logK = c .* cost_mat

    for _ in 1:num_iters
        # update f (row normalization) to match marginal mu
        S_f = g .+ logK                  # (1 × m) .+ (n × m) -> (n × m)
        lse_f = logsumexp(S_f; dims=2)
        f = log_mu .- lse_f

        # update g (column normalization) matches marginal nu
        S_g = f .+ logK                  # (n × 1) .+ (n × m) -> (n × m)
        lse_g = logsumexp(S_g; dims=1)
        g = log_nu .- lse_g
    end
    # transport plan: log_P = f + g + logK; -> (n, 1) .+ (1, m) .+ (n, m) -> (n, m)
    P = exp.(f .+ g .+ logK)
    return P
end

# compute pairwise squared Euclidean distance between two point clouds.
# xs: (d, n) - Latent points
# ys: (d, m) - Physical grid points
# return: (n, m) distance matrix
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
    pushforward_to_physical(measure::OrientedSurfaceMeasure, ot_plan::OptimalTransportPlan)

measure: (d × n) features on physical surface.
ot_plan:     (n × m) optimal transport plan from physical to latent spaces.
Returns point_cloud_transported: (d × m) features on latent grid.
"""
function pushforward_to_physical(
    measure::OrientedSurfaceMeasure{M},                    # (d × n)
    ot_plan::OptimalTransportPlan{M,G}                     # (n × m)
) where {M<:DenseMatrix{Float32},G<:LatentGrid}
    plan = ot_plan.plan
    points_transported = transport(measure.points, plan)   # (d × m)
    measure_transported = OrientedSurfaceMeasure(points_transported)
    return measure_transported
end

"""
    pullback_from_latent(measure::LatentOrientedSurfaceMeasure, ot_plan::OptimalTransportPlan)

measure: (d × m) features on latent grid.
ot_plan:     (n × m) optimal transport plan from physical to latent spaces.
Returns point_cloud_transported: (d × n) features on physical surface, i. e. what the latent
grid looks like after being bent/warped to match the surface of the input physical object.
"""
function pullback_from_latent(
    measure::LatentOrientedSurfaceMeasure{M},                           # (d × m)
    ot_plan::OptimalTransportPlan{M}                            # (n × m)
) where {M<:DenseMatrix{Float32}}
    plan = ot_plan.plan
    # transpose the transport plan to go from latent to physical
    points_transported = transport(measure.points, plan')   # (d × n)
    measure_transported = OrientedSurfaceMeasure(points_transported)
    return measure_transported
end

# for each point in the destination point cloud, find and assign the index of the closest
# point in the source point cloud
function assign_points(
    measure_src::AbstractMeasure,       # (d x n)
    measure_dst::AbstractMeasure        # (d x m)
)
    dists = pairwise_squared_euclidean_distance(measure_src.points, measure_dst.points)   # (n x m)
    index_pairs = vec(argmin(dists; dims=1))   # (m)
    indices_best = getindex.(index_pairs, 1)   # (m)
    return indices_best
end

function compute_encoder_and_decoder(
    measure::OrientedSurfaceMeasure{M},                               # (d × n)
    measure_latent::LatentOrientedSurfaceMeasure{M}                   # (d × m)
) where {M<:DenseMatrix{Float32}}
    ot_plan = OptimalTransportPlan(measure, measure_latent)           # (n × m)
    measure_transported = pushforward_to_physical(measure, ot_plan)   # (d × m)
    encoding_indices = assign_points(measure, measure_transported)    # (m)
    decoding_indices = assign_points(measure_transported, measure)    # (n)
    return (encoding_indices, decoding_indices, ot_plan, measure_transported)
end
