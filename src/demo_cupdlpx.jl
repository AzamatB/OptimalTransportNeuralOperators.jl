using cuPDLPx: LibcuPDLPx as Lib

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
