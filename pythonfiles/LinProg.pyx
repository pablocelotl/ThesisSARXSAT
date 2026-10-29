import cplex
import numpy as np
cimport numpy as np
DTYPE = np.int64
FTYPE=  np.double
ctypedef np.int64_t DTYPE_t
ctypedef np.double_t FTYPE_t
from cplex.exceptions import CplexError


cpdef populatebyrow(np.ndarray[FTYPE_t, ndim=2] output, np.ndarray[FTYPE_t, ndim=2] regressor, double delta, int ny,
                    int nu, int inputs, int outputs, slack):
    """Linear programming for checking feasible models
    
    Arguments:
        output {list} -- Set of outputs datapoints belonging to the specific interval that is being checked for feasiblity
        regressor {list} -- Set of regressor datapoints belonging to the specific interval that is being checked for feasiblity
        delta {double} -- Bound on the error
        ny {int} -- Order of the regressor
        nu {int} -- Order of the input
        inputs {int} -- Number of inputs
        outputs {int} -- Number of outputs

    Returns:
        variables {list} -- Variables of the minimization problem
        status {list} -- Status of the minimization problem
    """

    ## Without slack variables every output only constrains its own parameters, so the problem
    # splits into one independent LP per output: feasible exactly when all of them are.
    # With slack variables the outputs share the slacks, which does not fit in memory on large
    # problems, so those are split too, each output with its own slacks.
    cdef list sol_all
    cdef int o
    if outputs > 1 and (not slack or output.shape[1]*2*outputs > 100000):
        sol_all = []
        for o in range(outputs):
            (sol, status) = populatebyrow(output[o:o+1], regressor, delta, ny, nu, inputs, 1, slack)
            if status != 1:
                return([], 3)
            sol_all += sol
        return(sol_all, 1)

    ## Initializing the problem and turning of printing to screen
    prob = cplex.Cplex()
    prob.set_log_stream(None)
    prob.set_error_stream(None)
    prob.set_warning_stream(None)
    prob.set_results_stream(None)

    ## Type casting
    cdef int i, j, z
    cdef double x

    ## Defining parameters
    cdef list output_list=output.tolist()
    cdef int size=(len(output_list[0]))*2*outputs
    cdef int s=len(output_list[0])
    cdef int l=len(regressor[0])
    cdef int L=len(regressor[0])*outputs
    cdef list my_obj = [0.0]*L
    cdef list my_ub = [3.0]*L
    cdef list my_lb = [-3.0]*L
    cdef list my_colnames = ["x{0}".format(i) for i in range(1,L+1)]
    cdef list my_rhs = [x+1.0*delta for j in range(outputs) for x in output_list[j]]+[-x+1.0*delta for j in range(outputs) for x in output_list[j] ]
    cdef list my_rownames = ["c{0}".format(i) for i in range(1,size+1)]
    cdef list my_sense = ["L"] * size

    ## Turn on L1 minimization
    if slack:
        my_colnames=my_colnames+["s{0}".format(i) for i in range(1,s+1)]
        my_obj = my_obj+[10.0]*s
        my_ub = my_ub+[0.0]*s
        my_lb = my_lb+[-delta]*s

    ## Creating left side of constraints in sparse form (column indices + nonzero values):
    # first -regressor for every output, then +regressor for every output.
    # With slack variables, every row also gets -1 on the slack variable of its datapoint.
    cdef list rows=[]
    cdef list ind
    for cons in ((-regressor).tolist(), regressor.tolist()):
        for i in range(outputs):
            ind = list(range(i*l, (i+1)*l))
            for z in range(s):
                if slack:
                    rows.append(cplex.SparsePair(ind=ind+[L+z], val=cons[z]+[-1.0]))
                else:
                    rows.append(cplex.SparsePair(ind=ind, val=cons[z]))

    ## Add the required information to Cplex and solve the problem    
    prob.objective.set_sense(prob.objective.sense.minimize)
    prob.variables.add(obj=my_obj, lb=my_lb, ub=my_ub, names=my_colnames)
    prob.linear_constraints.add(lin_expr=rows, senses=my_sense,
                                rhs=my_rhs, names=my_rownames)
    ## On large problems CPLEX's default LP method runs out of memory, dual simplex does not.
    # Small problems keep the default so their results stay the same.
    if size > 100000:
        prob.parameters.lpmethod.set(prob.parameters.lpmethod.values.dual)
    prob.solve()

    ## Return the model and the status
    if prob.solution.get_status()==1:
        sol=prob.solution.get_values()
        sol=sol[0:L]
        sol=[-x for x in sol]
        return(sol,prob.solution.get_status())
    else:
        return([],3)

