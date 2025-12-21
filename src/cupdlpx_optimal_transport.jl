using cuPDLPx: LibcuPDLPx as Lib

function solve_optimal_transport_lp(
    costs::CuMatrix{Float32}, μ::Vector{Float64}, ν::Vector{Float64}
)
    (n, m) = Int32.(size(costs))
    @assert length(μ) == n
    @assert length(ν) == m
    num_vars = n * m
    num_cons = n + m

    # objective coefficients
    c = Vector{Float64}(undef, num_vars)
    copyto!(c, vec(costs))
    # constant term in objective
    c₀ = Float64[0.0]

    # equality constraints: con_lb == con_ub == [μ; ν]
    rhs = [μ; ν]
    con_lb = rhs
    con_ub = copy(rhs)

    # variable bounds: x >= 0
    var_lb = zeros(Float64, num_vars)
    var_ub = fill(Inf64, num_vars)

    # CSR structure: num_cons x num_vars, nnz = 2 * num_vars
    nnz = Int32(2 * num_vars)
    row_ptr = Vector{Int32}(undef, num_cons + 1)
    row_ptr[1] = one(Int32)
    @inbounds for i in 1:n
        row_ptr[i+1] = row_ptr[i] + m
    end
    @inbounds for j in 1:m
        row_ptr[n+j+1] = row_ptr[n+j] + n
    end

    col_ind = Vector{Int32}(undef, nnz)
    one_to_num_vars = Int32(1):Int32(num_vars)
    # source constraints block (rows 1..n): row-wise access in column-major flattening
    @. col_ind[one_to_num_vars] = linear_index(one_to_num_vars, n, m)
    # target constraints block (rows n+1..n+m): contiguous columns
    # col_ind[(num_vars+1):(2 * num_vars)] = 1:num_vars
    col_ind[(num_vars+1):end] .= one_to_num_vars

    # values are all ones
    vals = ones(Float64, nnz)
    x = solve_lp_problem(
        num_vars, num_cons, c, c₀, var_lb, var_ub, con_lb, con_ub, nnz, row_ptr, col_ind, vals
    )
    plan = extract_plan(x, n, m)
    return plan::CuMatrix{Float32}
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

function extract_plan(x::Vector{Float64}, n::Int, m::Int)
    x_gpu = CuVector{Float32}(x)
    plan = reshape(x_gpu, n, m)
    return plan
end

function solve_lp_problem(
    num_vars::Int32,
    num_cons::Int32,
    c::Vector{Float64},
    c₀::Vector{Float64},
    var_lb::Vector{Float64},
    var_ub::Vector{Float64},
    con_lb::Vector{Float64},
    con_ub::Vector{Float64},
    nnz::Int32,
    row_ptr::Vector{Int32},
    col_ind::Vector{Int32},
    vals::Vector{Float64}
)
    # form constraint matrix A in CSR format
    A_csr = Lib.MatrixCSR(nnz, pointer(row_ptr), pointer(col_ind), pointer(vals))

    # construct matrix_desc_t object for A
    # 1. allocate zeroed struct on Julia side
    A_desc_ref = Ref{Lib.matrix_desc_t}()
    A_desc_ref[] = Lib.matrix_desc_t(ntuple(_ -> UInt8(0), Val(56))) # clear memory
    A_desc_ptr = Base.unsafe_convert(Ptr{Lib.matrix_desc_t}, A_desc_ref)
    # 2. set scalar fields
    A_desc_ptr.m = num_cons
    A_desc_ptr.n = num_vars
    A_desc_ptr.fmt = Lib.matrix_csr
    A_desc_ptr.zero_tolerance = 0.0
    # 3. pass the CSR matrix A
    A_desc_ptr.data.csr = A_csr

    # set default PDHG parameters
    params_ref = Ref{Lib.pdhg_parameters_t}()
    params_ptr = Base.unsafe_convert(Ptr{Lib.pdhg_parameters_t}, params_ref)
    Lib.set_default_parameters(params_ptr)

    GC.@preserve c c₀ con_lb con_ub var_lb var_ub row_ptr col_ind vals A_desc_ref params_ref begin
        lp_problem = Lib.create_lp_problem(
            pointer(c),
            A_desc_ptr,
            pointer(con_lb),
            pointer(con_ub),
            pointer(var_lb),
            pointer(var_ub),
            pointer(c₀)
        )
        lp_problem == C_NULL && error("Failed to create LP problem")

        result_ptr = Lib.solve_lp_problem(lp_problem, params_ptr)
        result_ptr == C_NULL && error("Solver returned null result")

        result = unsafe_load(result_ptr)
        primal_solution = copy(
            unsafe_wrap(Array, result.primal_solution, Int(result.num_variables); own=false)
        )
        Lib.cupdlpx_result_free(result_ptr)
        Lib.lp_problem_free(lp_problem)
    end
    return primal_solution
end
