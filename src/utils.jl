function move_to_cpu(array::Array)
    return array
end
function move_to_cpu(array::CuArray{T,N}) where {T,N}
    return Array{T,N}(array)
end

function vec_type(::Type{<:Array{T}}) where {T}
    return Vector{T}
end
function vec_type(::Type{<:CuArray{T,N,M}}) where {T,N,M}
    return CuVector{T,M}
end

function estimate_plan_convergence(
    ot_plan::OptimalTransportPlan{M},                          # (n × m)
    measure::OrientedSurfaceMeasure{M},                        # (d × n)
    measure_l::LatentOrientedSurfaceMeasure{M}                 # (d × m)
) where {M<:DenseMatrix{Float32}}
    plan = ot_plan.plan                                        # (n × m)
    mu = measure.weights                                       # (n)
    nu = measure_l.weights                                     # (m)
    sparsity = count(iszero, plan) / length(plan)

    marginals_row = dropdims(sum(plan; dims=2); dims=2)        # (n)
    marginals_col = dropdims(sum(plan; dims=1); dims=1)        # (m)

    relative_errors_row = @. abs(marginals_row / mu - 1.0f0)   # (n)
    error_row = maximum(relative_errors_row)

    relative_errors_col = @. abs(marginals_col / nu - 1.0f0)   # (m)
    error_col = maximum(relative_errors_col)
    error = max(error_row, error_col)
    return (; error, sparsity)
end

function plot_ot_plan(
    ot_plan::OptimalTransportPlan;
    title::String="Optimal Transport Plan",
    xlabel::String="Source Points",
    ylabel::String="Target Points"
)
    plan = ot_plan.plan
    figure = Figure()
    # origin at top-left like matrix indexing
    axis = Axis(figure[1, 1]; title, xlabel, ylabel, yreversed=true)
    heatmap = heatmap!(axis, plan)
    Colorbar(figure[1, 2], heatmap)
    return figure
end

function save_sample(path::AbstractString, data_sample::OTEDataSample)
    open(path, "w") do io
        serialize(io, data_sample)
    end
    return nothing
end

function load_sample(path::AbstractString)
    data_sample = open(path, "r") do io
        deserialize(io)
    end
    return data_sample::OTEDataSample
end
