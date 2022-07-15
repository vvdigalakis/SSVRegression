using LinearAlgebra, JuMP, Gurobi#, CPLEX
include("utils.jl")

function sparse_regression(X, 
                           Y; 
                           sparsity::Int=min(5,size(X,3)),
                           lambda_reg::Real=size(X,1),
                           decrease_final_regularizer::Bool=true,
                           relaxation::Bool=false,
                           solver::Symbol=:Gurobi,
                           time_limit::Real=120.,
                           mip_gap::Real=1e-4,
                           nodefilestart::Real=Inf,
                           verbose::Int=0)
    
    lnr = SvarLearner()
    lnr.lnr_params[:sparsity] = sparsity
    lnr.lnr_params[:lambda_reg] = lambda_reg
    lnr.lnr_params[:relaxation] = relaxation
    lnr.lnr_params[:solver] = solver
    lnr.lnr_params[:time_limit] = time_limit
    lnr.lnr_params[:mip_gap] = mip_gap
    
    N,T,D = size(X)
    X_all = zeros(N*T,D)
    Y_all = zeros(N*T)
    i = 1
    for n=1:N, t=1:T
        X_all[i,:] .= X[n,t,:]
        Y_all[i] = Y[n,t]
        i += 1
    end
                        
    t_total = @elapsed Sparse_Regressor = oa_formulation(
                  X_all, Y_all, sparsity, 1/lambda_reg, decrease_final_regularizer=decrease_final_regularizer,
                  ΔT_max=time_limit, verbose=verbose, Gap=mip_gap, solver=solver, nodefilestart=nodefilestart
                )
    indices, w, solverTime, status, Gap, cutCount, cutTime = Sparse_Regressor
    z = zeros(T,D)
    beta = zeros(T,D)
    for i in indices
        i_index = [j for (j,ind) in enumerate(indices) if i==ind][1]
        z[:,i] .= 1
        beta[:,i] .= w[i_index]
    end
        
    lnr.beta, lnr.z, lnr.t = beta, z, t_total
    lnr.lnr_stats[:status] = status
    lnr.lnr_stats[:gap] = Gap
    lnr.lnr_stats[:t_solver] = solverTime
    lnr.lnr_stats[:cut_count] = cutCount
    lnr.lnr_stats[:t_cut_total] = cutTime
    lnr.lnr_stats[:t_cut_avg] = cutTime/cutCount
    
    GC.gc()
                    
    return lnr
end              
                
#############   
# SR MIO
#############

function oa_formulation(X, Y, k::Int, γ;
          decrease_final_regularizer::Bool=false,
          indices0=findall(rand(size(X,2)) .< k/size(X,2)),
          ΔT_max=60, verbose=false, Gap=0e-3, solver::Symbol=:Gurobi,  nodefilestart::Real=Inf)

    n,p = size(X)
                                    
    miop = (solver == :Gurobi) ? Model(Gurobi.Optimizer) : Model(CPLEX.Optimizer)
    set_optimizer_attribute(miop, (solver == :Gurobi) ? "TimeLimit" : "CPX_PARAM_TILIM", ΔT_max)
    set_optimizer_attribute(miop, (solver == :Gurobi) ? "OutputFlag" : "CPX_PARAM_SCRIND", 1*verbose)
    set_optimizer_attribute(miop, (solver == :Gurobi) ? "MIPGap" : "CPX_PARAM_EPGAP", Gap)
    set_optimizer_attribute(miop, (solver == :Gurobi) ? "Threads" : "CPXPARAM_Threads", 1)
    if (solver == :Gurobi) set_optimizer_attribute(miop, "NodefileStart", nodefilestart) end

    s0 = zeros(p); s0[indices0] .= 1.
    firstCutTime = @elapsed c0, ∇c0 = inner_op(X, Y, s0, γ)

    # Optimization variables
    @variable(miop, s[j=1:p], Bin, start=s0[j])
    @variable(miop, t>=0, start=1.005*c0)

    for j in 1:p
        JuMP.set_start_value(s[j], s0[j])
    end
    JuMP.set_start_value(t, 1.005*c0)

    # Objective
    @objective(miop, Min, t)

    # Constraints
    @constraint(miop, sum(s) <= k)

    #Root node analysis
    cutCount=1
    cutTime = firstCutTime
    @constraint(miop, t>= c0 + dot(∇c0, s-s0))

    # Outer approximation method for Convex Integer Optimization (CIO)
    function outer_approximation(cb_data)
        s_val = [callback_value(cb_data, s[j]) for j in 1:p] 
        s_val = 1.0 .* (rand(p) .< s_val) # JuMP updates calls Lazy Callbacks at fractional solutions as well
        newCutTime = @elapsed c, ∇c = inner_op(X, Y, s_val, γ)
        con = @build_constraint(t >= c + dot(∇c, s-s_val))
        MOI.submit(miop, MOI.LazyConstraint(cb_data), con)
        cutCount += 1
        cutTime += newCutTime
    end
    MOI.set(miop, MOI.LazyConstraintCallback(), outer_approximation)

    mem = @allocated optimize!(miop)
    if @isdefined DBG
        s_dbg = "Mem during solver: $(mem/1024^2)"
        debug(DBG, :memory, s_dbg)
    end
                                        
    if has_values(miop)
        status = termination_status(miop)
        Δt = JuMP.solve_time(miop)
        Gap = 1 - JuMP.objective_bound(miop) /  abs(JuMP.objective_value(miop))
        s_opt = value.(s)
    else
        status = nothing
        Δt = JuMP.solve_time(miop)
        Gap = 0
        s_opt = zeros(p)
    end

    # Find selected regressors and run a standard linear regression with Tikhonov regularization
    indices = findall(s_opt .> .5)
    if decrease_final_regularizer
        γ = sqrt(γ)
    end
    w = recover_primal(X[:, indices], Y, γ)
    
    miop = nothing
    GC.gc()

    return indices, w, Δt, status, Gap, cutCount, cutTime
end

                                                                    
###
# Inner problem
###

function inner_op(X, Y, s, γ)
  indices = findall(s .> .5); k = length(indices)
  n,p = size(X)

  # Compute optimal dual parameter
  α = sparse_inverse(X[:, indices], Y, γ)
  c = value_dual(X, Y, α, indices, k, γ)

  ∇c = zeros(p)
  for j in 1:p
    ∇c[j] = -γ/2*dot(X[:,j],α)^2
  end
  return c, ∇c
end 
                                                                    
function recover_primal(Z, Y, γ)
  CM = Matrix(I, size(Z,2), size(Z,2))/γ + Z'*Z      # The capacitance matrix
  α = -Y + Z*(CM\(Z'*Y))            # Matrix Inversion Lemma
  return -γ*Z'*α                    # Regressor
end                 
                
function sparse_inverse(X, Y, γ)

  n = size(X, 1)
  k = size(X, 2)

  CM = Matrix{Float64}(I,k,k)/γ + X'*X      # The capacitance matrix
  α = -Y + X*(CM\(X'*Y))       # Matrix Inversion Lemma

  return α
end
                                    
##Dual objective function value for a given dual variable α
function value_dual(X, Y, α, indices, n_indices, γ)
  v = - sum([fenchel(Y[i], α[i]) for i in 1:size(X, 1)])
  for j in 1:n_indices
    v -= γ/2*(dot(X[:, indices[j]], α)^2)
  end
  return v
end
                                    
##Point-wise value of the Fenchel conjugate for each loss function
function fenchel(y, a)
  return .5*a^2 + a*y
end