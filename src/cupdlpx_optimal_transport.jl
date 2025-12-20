using CUDA
using CUDA.CUSPARSE

function form_optimal_transport_lp(
    cost::CuMatrix{Float32}, μ::CuVector{Float32}, ν::CuVector{Float32}
)
    (n, m) = Int32.(size(cost))
    @assert length(μ) == n
    @assert length(ν) == m
    num_vars = n * m
    num_cons = n + m
    nnz = 2 * num_vars

    # objective coefficients
    c = vec(cost)  # CuVector view, no copy

    # CSR structure: num_cons x num_vars, nnz = 2 * num_vars
    # row pointers on CPU (tiny), then copy to GPU
    rowptr = Vector{Int32}(undef, num_cons + 1)
    rowptr[1] = one(Int32)
    @inbounds for i in 1:n
        rowptr[i+1] = rowptr[i] + m
    end
    @inbounds for j in 1:m
        rowptr[n+j+1] = rowptr[n+j] + n
    end

    colval = CuVector{Int32}(undef, nnz)
    one_to_num_vars = CuVector(Int32(1):Int32(num_vars))
    # source constraints block (rows 1..n): row-wise access in column-major flattening
    @. colval[one_to_num_vars] = linear_index(one_to_num_vars, n, m)
    # target constraints block (rows n+1..n+m): contiguous columns
    # colval[(num_vars+1):(2 * num_vars)] = 1:num_vars
    colval[(num_vars+1):end] .= one_to_num_vars

    # values are all ones
    nzval = CUDA.ones(Float32, nnz)
    A = CuSparseMatrixCSR(CuArray(rowptr), colval, nzval, (num_cons, num_vars))

    # equality constraints: con_lb == con_ub == [μ; ν]
    rhs = vcat(μ, ν)
    con_lb = rhs
    con_ub = copy(rhs)

    # variable bounds: x >= 0
    var_lb = CUDA.zeros(Float32, num_vars)
    var_ub = CUDA.fill(Inf32, num_vars)
    lp_probelm = cuPDLPx.LinearProgrammingProblem(c, A, con_lb, con_ub, var_lb, var_ub)
    return lp_probelm
end

function linear_index(k::Int32, n::Int32, m::Int32)
    uno = one(Int32)
    km1 = k - uno
    # im1 = i-1   (0-based row index in row-major enumeration)
    im1 = km1 ÷ m
    # jm1 = j-1   (0-based col index)
    jm1 = km1 - im1 * m
    # 1-based column-major linear index: i + (j-1)*n
    col_index = im1 + uno + jm1 * n
    return col_index
end

function extract_plan(x::CuVector{Float32}, n::Int, m::Int)
    return reshape(x, n, m)
end
