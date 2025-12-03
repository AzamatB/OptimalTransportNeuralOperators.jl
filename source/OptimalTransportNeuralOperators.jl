using GeometryBasics
using FileIO
using MeshIO
using NNlib
using Statistics

# extract vertices from a mesh as a 3×n Matrix{T}
function extract_vertices(mesh::Mesh)
    T = Float32
    vertices = mesh.vertex_attributes.position
    num_points = length(vertices)
    dim = length(first(vertices))
    points = Matrix{T}(undef, dim, num_points)
    eachcol(points) .= vertices
    return points::Matrix{T}
end

# Create a regular nθ×nφ grid on a torus with major radius R and minor radius r.
# Returns (points_flat, θ_vals, φ_vals), where points_flat is (n, 3), n = nθ*nφ.
# As θ changes from 0 to 2π, a point traces out a full circle around the z-axis,
# forming the outer "ring" of the torus.
# As φ changes from 0 to 2π, a point moves around the cross-sectional circle of the tube.
function generate_torus_latent_points(R::Real, r::Real, nθ::Int, nφ::Int)
    @assert R > r > 0
    T = Float32
    radius_maj = T(R)
    radius_min = T(r)
    θ_vals = collect(range(T(0), T(2π), length=nθ + 1)[1:end-1])
    φ_vals = collect(range(T(0), T(2π), length=nφ + 1)[1:end-1])

    points = Array{T,3}(undef, 3, nθ, nφ)
    @inbounds for j in eachindex(φ_vals)
        φ = φ_vals[j]
        v = radius_maj + radius_min * cos(φ)
        z = radius_min * sin(φ)
        for i in eachindex(θ_vals)
            θ = θ_vals[i]
            x = v * cos(θ)
            y = v * sin(θ)
            points[1, i, j] = x
            points[2, i, j] = y
            points[3, i, j] = z
        end
    end
    # 3 × n (column-major, flattens θ first, then φ)
    points_flat = reshape(points, 3, :)
    return points_flat::Matrix{T}, θ_vals::Vector{T}, φ_vals::Vector{T}
end

# Computes pairwise squared Euclidean distance between two point clouds.
# xs: (d, n) - Physical points
# ys: (d, m) - Latent grid points
# Returns: (n, m) distance matrix
function pairwise_squared_euclidean_distance(xs::DenseMatrix{Float32}, ys::DenseMatrix{Float32})
    # xᵀx, yᵀy
    xs_sq = sum(abs2, xs; dims=1)  # (1, n)
    ys_sq = sum(abs2, ys; dims=1)  # (1, m)
    xs_sqᵀ = xs_sq'  # (n, 1)
    # xᵀy
    xys = xs' * ys  # (n, m)
    # xᵀx + yᵀy - 2xᵀy,  ∀ x ∈ xs, y ∈ ys
    dists = @. xs_sqᵀ + ys_sq - 2 * xys
    return dists
end

"""
    sinkhorn_plan(xs::M, ys::M, num_iters::Int=50)

    Compute an entropic regularized optimal transport plan between discrete measures on
xs (d×n) and ys (d×m) using the Log-Domain Sinkhorn-Knopp algorithm, assuming uniform marginals.
    Returns P (n×m) such that P >= 0.
"""
function sinkhorn_plan(xs::M, ys::M, num_iters::Int=50) where {M<:DenseMatrix{Float32}}
    T = Float32
    ε = T(0.01)
    cost_mat = pairwise_squared_euclidean_distance(xs, ys)
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

    # kernel in log domain: K = -cost_mat / (ε * mean(cost_mat))
    c = -ε * mean(cost_mat) # factor in normalization by mean cost for numerical stability
    logK = cost_mat ./ c

    for _ in 1:num_iters
        # update f (row normalization):     S_f = g + logK -> (1, m) .+ (n, m) -> (n, m)
        S_f = g .+ logK
        lse_f = logsumexp(S_f; dims=2)
        f = log_mu .- lse_f
        # update g (column normalization):  S_g = f + logK -> (n, 1) .+ (n, m) -> (n, m)
        S_g = f .+ logK
        lse_g = logsumexp(S_g; dims=1)
        g = log_nu .- lse_g
    end
    # final row normalization
    S_f = g .+ logK
    lse_f = logsumexp(S_f; dims=2)
    f = log_mu .- lse_f

    # transport plan: log_P = f + g + logK; -> (n, 1) .+ (1, m) .+ (n, m) -> (n, m)
    P = exp.(f .+ g .+ logK)
    return P
end
