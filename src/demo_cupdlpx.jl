using cuPDLPx

const Lib = cuPDLPx.LibcuPDLPx

# Maximize x + y + 2z
# s.t. x + 2y + 3z <= 4
#      x + y >= 1
#      0 <= x, y, z <= 1
n_vars = 3
m_cons = 2

# cuPDLPx solves min problems, so negate the objective.
c = Cdouble[-1.0, -1.0, -2.0]
obj_const = Cdouble[0.0]

var_lb = Cdouble[0.0, 0.0, 0.0]
var_ub = Cdouble[1.0, 1.0, 1.0]

# CSR matrix with 0-based column indices.
# Row 0:  1.0,  2.0, 3.0
# Row 1: -1.0, -1.0, 0.0  (represents x + y >= 1 as -x - y <= -1)
csr_vals = Cdouble[1.0, 2.0, 3.0, -1.0, -1.0]
csr_col_ind = Cint[0, 1, 2, 0, 1]
csr_row_ptr = Cint[0, 3, 5]
nnz = Cint(5)

con_lb = Cdouble[-Inf, -Inf]
con_ub = Cdouble[4.0, -1.0]

A_csr = Lib.MatrixCSR(
    nnz,
    pointer(csr_row_ptr),
    pointer(csr_col_ind),
    pointer(csr_vals)
)

A_desc_ref = Ref{Lib.matrix_desc_t}(Lib.matrix_desc_t(ntuple(_ -> UInt8(0), 56)))
A_desc_ptr = Base.unsafe_convert(Ptr{Lib.matrix_desc_t}, A_desc_ref)
A_desc_ptr.m = Cint(m_cons)
A_desc_ptr.n = Cint(n_vars)
A_desc_ptr.fmt = Lib.matrix_csr
A_desc_ptr.zero_tolerance = 0.0
A_desc_ptr.data.csr = A_csr

params_ref = Ref{Lib.pdhg_parameters_t}()
params_ptr = Base.unsafe_convert(Ptr{Lib.pdhg_parameters_t}, params_ref)
Lib.set_default_parameters(params_ptr)

GC.@preserve c obj_const var_lb var_ub csr_vals csr_col_ind csr_row_ptr con_lb con_ub A_desc_ref params_ref begin
    prob = Lib.create_lp_problem(
        pointer(c),
        A_desc_ptr,
        pointer(con_lb),
        pointer(con_ub),
        pointer(var_lb),
        pointer(var_ub),
        pointer(obj_const)
    )
    prob == C_NULL && error("Failed to create LP problem")

    result_ptr = Lib.solve_lp_problem(prob, params_ptr)
    result_ptr == C_NULL && error("Solver returned null result")

    result = unsafe_load(result_ptr)
    primal = copy(unsafe_wrap(Array, result.primal_solution, Int(result.num_variables); own=false))
    dual = copy(unsafe_wrap(Array, result.dual_solution, Int(result.num_constraints); own=false))

    println("Termination reason: ", result.termination_reason)
    println("Primal objective (min): ", result.primal_objective_value)
    println("Primal objective (max): ", -result.primal_objective_value)
    println("Primal solution: ", primal)
    println("Dual solution: ", dual)

    Lib.cupdlpx_result_free(result_ptr)
    Lib.lp_problem_free(prob)
end
