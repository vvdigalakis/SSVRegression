using LinearAlgebra, StatsBase, JuMP, Gurobi, Ipopt#, CPLEX
include("utils.jl")

function regression_sum_of_norms(
        X, # array of dimension N*T*D
        Y; # array of dimension N*T
        edges::Dict{Int, Vector{Int}}=Dict(
           t => t==1 ? (size(X,2)==1 ? Int[] : [t+1]) : (t==size(X,2) ? [t-1] : [t-1,t+1]) for t=1:size(X,2)
        ), # defaults to temporally varying model
        lambda_svar::Real=sqrt(size(X,1)), # regularization weight
        norm::Symbol=:l1, # l1 or l2 norm
        penalized::Bool=true, # penalized or constrained approach
        multiple_constr::Bool=true, # use a different constraint per time step (in case constrained=true)
        lasso::Bool=true,
        lambda_reg::Real=size(X,1), # regularization weight
        error_term::Symbol=:l2,
        solver::Symbol=:Gurobi,
        time_limit::Real=120.,
        nodefilestart::Real=Inf,
        verbose=0
    )
    
    lnr = SvarLearner()
    lnr.lnr_params[:edges] = edges
    lnr.lnr_params[:lambda_svar] = lambda_svar
    lnr.lnr_params[:norm] = norm
    lnr.lnr_params[:penalized] = penalized
    lnr.lnr_params[:multiple_constr] = multiple_constr
    lnr.lnr_params[:lasso] = lasso
    lnr.lnr_params[:lambda_reg] = lambda_reg
    lnr.lnr_params[:error_term] = error_term
    lnr.lnr_params[:solver] = solver
    lnr.lnr_params[:time_limit] = time_limit

    @assert norm in [:l1,:l2] "Undefined norm for regression_sum_of_norms."
    
    N,T,D = size(X)
    
    t0 = time()
        
    # Create model
    if (solver == :Gurobi)
        m = Model(Gurobi.Optimizer)
        set_optimizer_attribute(m, "TimeLimit", time_limit)
        set_optimizer_attribute(m, "OutputFlag", 1*verbose)
        set_optimizer_attribute(m, "Threads" , 1)
        set_optimizer_attribute(m, "NodefileStart", nodefilestart) 
        set_optimizer_attribute(m, "NonConvex", 2) 
        
    elseif (solver == :CPLEX) 
        m = Model(CPLEX.Optimizer)
        set_optimizer_attribute(m, "CPX_PARAM_TILIM", time_limit)
        set_optimizer_attribute(m, "CPX_PARAM_SCRIND", 1*verbose)
        set_optimizer_attribute(m, "CPXPARAM_Threads", 1)
        
    elseif (solver == :Ipopt) 
        m = Model(Ipopt.Optimizer)
        set_optimizer_attribute(m, "max_cpu_time", time_limit)
        if verbose == 0
           set_silent(m)
        end
        
    end
    
    @variable(m, beta[1:T,1:D])
    
    if (!penalized) & (T>1)
        add_svar_constr!(m,beta,edges,lambda_svar,norm,multiple_constr)
    end
    
    add_objective!(m, X, Y, beta, edges, lambda_svar, norm, penalized, lambda_reg, lasso, error_term)
    
    mem = @allocated optimize!(m)
    if @isdefined DBG
        s = "Mem during solver: $(mem/1024^2)"
        debug(DBG, :memory, s)
    end
    
    if has_values(m)
        status = termination_status(m)
        t_solver = JuMP.solve_time(m)
        beta_opt = JuMP.value.(beta)
        obj_opt = JuMP.objective_value(m)
    else
        status = nothing
        t_solver = JuMP.solve_time(m)
        beta_opt = zeros(T,D)
        obj_opt = -1
    end
        
    beta_opt[abs.(beta_opt) .< 1e-9] .= 0
    z_opt = zeros(T,D)
    z_opt[abs.(beta_opt).>0] .= 1
    sparsity = maximum([sum(z_opt[t,:]) for t=1:T])
    global_sparsity = sum([sum(z_opt[:,d]) for d=1:D] .> 0.5)
    
    t_total = time() - t0
    
    lnr.beta, lnr.z, lnr.t = beta_opt, z_opt, t_total
    lnr.lnr_stats[:obj] = obj_opt
    lnr.lnr_stats[:status] = status
    lnr.lnr_stats[:t_solver] = t_solver
    
    m = nothing
    GC.gc()
    
    return lnr 
end

function add_svar_constr!(m,beta,edges,lambda_svar,norm,multiple_constr)
    
    T,D = size(beta)
    
    if (norm==:l1) & multiple_constr
        @variable(m, w_constr[t=1:T,t_adj in edges[t],d=1:D; t<t_adj])
        @constraint(m, [t=1:T,t_adj in edges[t],d=1:D; t<t_adj], w_constr[t,t_adj,d]>=beta[t,d]-beta[t_adj,d])
        @constraint(m, [t=1:T,t_adj in edges[t],d=1:D; t<t_adj], w_constr[t,t_adj,d]>=-beta[t,d]+beta[t_adj,d])
        @constraint(m, [t=1:T,t_adj in edges[t]; t<t_adj], sum(w_constr[t,t_adj,:]) <= lambda_svar )
        
    elseif (norm==:l1) & !multiple_constr
        @variable(m, w_constr[t=1:T,t_adj in edges[t],d=1:D; t<t_adj])
        @constraint(m, [t=1:T,t_adj in edges[t],d=1:D; t<t_adj], w_constr[t,t_adj,d]>=beta[t,d]-beta[t_adj,d])
        @constraint(m, [t=1:T,t_adj in edges[t],d=1:D; t<t_adj], w_constr[t,t_adj,d]>=-beta[t,d]+beta[t_adj,d])
        @constraint(m, sum(w_constr) <= lambda_svar )

    elseif (norm==:l2) & multiple_constr
        @constraint(m, [t=1:T,t_adj in edges[t]; t<t_adj], sum((beta[t,:] .- beta[t_adj,:]).^2) <= lambda_svar )
        
    elseif (norm==:l2) & !multiple_constr
        @constraint(m, sum( sum((beta[t,:] .- beta[t_adj,:]).^2) for t=1:T for t_adj in edges[t] if t<t_adj ) <= lambda_svar )
    
    end
    
end

function add_objective!(m, X, Y, beta, edges, lambda_svar, norm, penalized, lambda_reg, lasso, error_term)
    
    N,T,D = size(X)
    
    if error_term == :l2
        err = sum(sum((Y[n,t] - sum(beta[t,:].*X[n,t,:]))^2 for t=1:T) for n=1:N)
    elseif error_term == :l1
        @variable(m, w_err[n=1:N,t=1:T] >= 0)
        @constraint(m, [n=1:N,t=1:T], w_err[n,t]>=Y[n,t] - sum(beta[t,:].*X[n,t,:]))
        @constraint(m, [n=1:N,t=1:T], w_err[n,t]>=-(Y[n,t] - sum(beta[t,:].*X[n,t,:])))
        err = sum(w_err)
    else
        error("Unsupported error term.")
    end
    
    # l1 SV penalty
    if (norm == :l1) & (T>1)
        @variable(m, w_pen[t=1:T,t_adj in edges[t],d=1:D; t<t_adj] >= 0)
        @constraint(m, [t=1:T,t_adj in edges[t],d=1:D; t<t_adj], w_pen[t,t_adj,d]>=beta[t,d]-beta[t_adj,d])
        @constraint(m, [t=1:T,t_adj in edges[t],d=1:D; t<t_adj], w_pen[t,t_adj,d]>=-beta[t,d]+beta[t_adj,d])
        svar = sum(w_pen)
                            
    # l2 SV penalty (SUM OF NORMS)
    elseif (norm == :l2) & (T>1) 
        @variable(m, svar_edge[t=1:T,t_adj in edges[t]; t<t_adj] >= 0)
        @constraint(m, [t=1:T,t_adj in edges[t],d=1:D; t<t_adj], [ svar_edge[t,t_adj] ; (beta[t,:] - beta[t_adj,:]) ] in SecondOrderCone() )
        @variable(m, svar >= 0)
        @constraint(m, svar >= sum(svar_edge))                            
                            
    else
        svar = 0 
    end
    
    if lasso
        @variable(m, w_lasso[t=1:T,d=1:D] >= 0)
        @constraint(m, [t=1:T,d=1:D], w_lasso[t,d]>=beta[t,d])
        @constraint(m, [t=1:T,d=1:D], w_lasso[t,d]>=-beta[t,d])
        reg = sum(w_lasso)
    else
        reg = 0
    end
    
    @objective(m, Min, err + lambda_svar*svar + lambda_reg*reg)

end