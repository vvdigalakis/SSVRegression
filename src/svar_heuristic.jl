using LinearAlgebra, JuMP, Gurobi#, CPLEX
include("utils.jl")

#################
# Heuristic: separable in variables
#################

function svar_heuristic(
        X, # array of dimension N*T*D
        Y; # array of dimension N*T
        edges::Dict{Int, Vector{Int}}=Dict(
           t => t==1 ? (size(X,2)==1 ? Int[] : [t+1]) : (t==size(X,2) ? [t-1] : [t-1,t+1]) for t=1:size(X,2)
        ), # defaults to temporally varying model
        sparsity::Int=min(5,size(X,3)),
        global_sparsity::Int=min(convert(Int,ceil(1.5*sparsity)),size(X,3)),
        global_sparsity_relative::Int=-1,
        sparsely_varying::Int=2*(global_sparsity-sparsity),
        lambda_reg::Real=size(X,1),
        lambda_svar::Real=sqrt(size(X,1)),
        decrease_final_regularizer::Bool=true,
        solver::Symbol=:Gurobi,
        time_limit::Real=120.,
        mip_gap::Real=1e-4,
        nodefilestart::Real=Inf,
        relax::Bool=true,
        round::Bool=true,
        verbose::Int=0
    )
    
    # Express global sparsity as relative increase wrt local (makes cross validation implementation easier)
    if global_sparsity_relative >= 0
        global_sparsity = sparsity + global_sparsity_relative
    end
    
    lnr = SvarLearner()
    lnr.lnr_params[:edges] = edges
    lnr.lnr_params[:sparsity] = sparsity
    lnr.lnr_params[:global_sparsity] = global_sparsity
    lnr.lnr_params[:sparsely_varying] = sparsely_varying
    lnr.lnr_params[:lambda_reg] = lambda_reg
    lnr.lnr_params[:lambda_svar] = lambda_svar
    lnr.lnr_params[:solver] = solver
    lnr.lnr_params[:time_limit] = time_limit
    lnr.lnr_params[:mip_gap] = mip_gap
    lnr.lnr_params[:relax] = relax
    lnr.lnr_params[:round] = round
    
    N,T,D = size(X)
    
    if (@isdefined DBG) & (verbose>0)
        s = "Starting heuristic: N,T,D,sparsity = $((N,T,D,sparsity))"
        debug(DBG, :progress, s)
    end
    
    # Compute univariate regressions
    t0 = time()
    L = zeros(T,D)
    for t=1:T, d=1:D
        reg = lambda_reg+lambda_svar*length(edges[t])
        beta_td = sum(X[:,t,d].*Y[:,t])/(sum(X[:,t,d].^2)+reg)
        L[t,d] = sum((Y[:,t] .- X[:,t,d].*beta_td).^2) + reg*beta_td^2
    end
    t_preprocess = time()-t0
    
    # Create model
    mio = (solver == :Gurobi) ? Model(Gurobi.Optimizer) : Model(CPLEX.Optimizer)
    set_optimizer_attribute(mio, (solver == :Gurobi) ? "TimeLimit" : "CPX_PARAM_TILIM", time_limit)
    set_optimizer_attribute(mio, (solver == :Gurobi) ? "OutputFlag" : "CPX_PARAM_SCRIND", 1*verbose)
    set_optimizer_attribute(mio, (solver == :Gurobi) ? "MIPGap" : "CPX_PARAM_EPGAP", mip_gap)
    set_optimizer_attribute(mio, (solver == :Gurobi) ? "Threads" : "CPXPARAM_Threads", 1)
    if (solver == :Gurobi) set_optimizer_attribute(mio, "NodefileStart", nodefilestart) end
    
    # Variables
    if relax
        @variable(mio, 0<=z[t=1:T,d=1:D]<=1)
        @variable(mio, 0<=z_overall[d=1:D]<=1)
        if T>1
            @variable(mio, 0<=z_diff[t=1:T,t_adj in edges[t],d=1:D; t<t_adj]<=1)
        end
    else
        @variable(mio, z[t=1:T,d=1:D], Bin)
        @variable(mio, z_overall[d=1:D], Bin)
        if T>1
            @variable(mio, z_diff[t=1:T,t_adj in edges[t],d=1:D; t<t_adj], Bin)
        end
    end

    @objective(mio,Min,sum(L.*z)/D)

    # Sparsity constraints
    @constraint(mio, [t=1:T], sum(z[t,:]) == sparsity)
    @constraint(mio, [t=1:T,d=1:D], z[t,d] <= z_overall[d])
    @constraint(mio, sum(z_overall) <= global_sparsity)
    if T>1
        @constraint(mio, [t=1:T,t_adj in edges[t],d=1:D; t<t_adj], z_diff[t,t_adj,d] >= z[t,d]-z[t_adj,d])
        @constraint(mio, [t=1:T,t_adj in edges[t],d=1:D; t<t_adj], z_diff[t,t_adj,d] >= -z[t,d]+z[t_adj,d])
        @constraint(mio, sum(z_diff) <= sparsely_varying)
    end
    
    mem = @allocated optimize!(mio)
    if (@isdefined DBG) & (verbose>0)
        s = "Mem during solver: $(mem/1024^2)"
        debug(DBG, :memory, s)
    end
    
    if has_values(mio)
        status = termination_status(mio)
        t_solver = JuMP.solve_time(mio)
        gap = 1 - JuMP.objective_bound(mio) /  abs(JuMP.objective_value(mio))
        z_heuristic = JuMP.value.(z)
    else
        status = nothing
        t_solver = time_limit
        gap = -1
        z_heuristic =  zeros(T,D)
    end
    
    if (@isdefined DBG) & (verbose>0)
        s = "Solver done! Perform rounding? $round"
        debug(DBG, :progress, s)
    end
     
    # Perform rounding procedure
    if round
        z_heuristic = perform_rounding(z_heuristic, L, edges, sparsity, global_sparsity, sparsely_varying, verbose=verbose)
    end
    
    # Find global support
    S = []
    for t=1:T
        append!(S,[i for i=1:D if (z_heuristic[t,i] > 1e-3)])
    end
    S = sort(unique(S))
    
    if (@isdefined DBG) & (verbose>0)
        s = "Estimating final coefs"
        debug(DBG, :progress, s)
    end
         
    # Compute constants (for solving inner problem given estimated support)
    t_constant_computation = @elapsed M, m, offset = compute_M(
        X[:,:,S],
        Y,edges,lambda_reg,lambda_svar,verbose=verbose
    )
    z_heuristic_sliced = z_heuristic[:,S]
    
    # Solve inner problem
    if decrease_final_regularizer
        lambda_reg = sqrt(lambda_reg)
    end
    t_inner_problem = @elapsed beta_heuristic_small,obj_heuristic,_ = inner_problem(
                        M,m,offset,edges,z_heuristic_sliced,lambda_reg,lambda_svar,
                        compute_gradient=false,verbose=verbose
                    )
                    
    beta_heuristic = zeros(T,D)
    for t=1:T, (d_i,d) in enumerate(S)
        beta_heuristic[t,d] = beta_heuristic_small[t,d_i]
    end
    
    t_total = time()-t0
    
    lnr.beta, lnr.z, lnr.t = beta_heuristic, z_heuristic, t_total
    lnr.lnr_stats[:obj] = obj_heuristic
    lnr.lnr_stats[:status] = status
    lnr.lnr_stats[:gap] = gap
    lnr.lnr_stats[:t_preprocess] = t_preprocess
    lnr.lnr_stats[:t_solver] = t_solver
    lnr.lnr_stats[:t_constant_computation] = t_constant_computation
    lnr.lnr_stats[:t_inner_problem] = t_inner_problem
    
    mio = nothing
    GC.gc()
                    
    return lnr
 
end
   
                    
###
# Auxiliary functions
###
                    
function perform_rounding(z, L, edges, sparsity, global_sparsity, sparsely_varying; verbose=0)
                        
    T,D = size(z)

    # Find indices of ones or non-integral variables
    indices_nonz = findall((z .> 1e-3))
    feature_nonz = unique(map(x -> x[2], indices_nonz))
                        
    # Set all such variables to 1 
    z[indices_nonz] .= 1
    z[Not(indices_nonz)] .= 0
                        
    # Remove such variables based on initial loss estimate
    L_nonz = mean(L[feature_nonz], dims=2)
    feature_nonz_removal_order = feature_nonz[sortperm(L_nonz, rev=false)[:,1]]
        
    if (@isdefined DBG) & (verbose>0)
        s = "Nonzero feature removal order: $(feature_nonz_removal_order)"
        debug(DBG, :progress, s)
    end
                        
    # While infeasible, keep removing
    while !(
            check_local_sparsity(z, sparsity)
            & check_global_sparsity(z, global_sparsity)
            & check_sparsely_varying(z, edges, sparsely_varying)
        )
    
        if (@isdefined DBG) & (verbose>0)
            s = "Sparsity params: $(eval_sparsity(z, edges))"
            debug(DBG, :progress, s)
        end
                            
        # Remove feature corresponding to higher loss
        remove_ind = pop!(feature_nonz_removal_order)
        z[:,remove_ind] .= 0  
    
        if (@isdefined DBG) & (verbose>0)
            s = "Removing feature $(remove_ind)"
            debug(DBG, :progress, s)
        end
                                    
    end
                        
    return z
                        
end   
                    
# function perform_rounding(z, L, edges, sparsity, global_sparsity, sparsely_varying)

#     # Find indices of non-integral variables
#     indices_nonint = findall((z .* (1 .- z) .> 0))
#     indices_ones = findall((z .> 1-1e-3))
                        
#     # Set all such variables to 1 
#     z[indices_nonint] .= 1
#     z[indices_ones] .= 1
                        
#     # Remove such variables based on initial loss estimate
#     indices_nonint_removal_order = sortperm(L[indices_nonint], rev=false)
#     indices_ones_removal_order = sortperm(L[indices_ones], rev=false)
                        
#     # While infeasible, keep removing
#     while !(
#             check_local_sparsity(z, sparsity)
#             & check_global_sparsity(z, global_sparsity)
#             & check_sparsely_varying(z, edges, sparsely_varying)
#         )
#         # Remove index corresponding to higher loss
#         if length(indices_nonint_removal_order) > 0
#             remove_ind = pop!(indices_nonint_removal_order)
#             z[indices_nonint[remove_ind]] = 0
#         else
#             remove_ind = pop!(indices_ones_removal_order)
#             z[indices_ones[remove_ind]] = 0
#         end
#     end
                        
#     return z
                        
# end      