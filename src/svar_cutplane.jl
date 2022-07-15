using SparseArrays, LinearAlgebra, StatsBase, JuMP, Gurobi#, CPLEX
include("utils.jl")

#################
# MIO formulation
#################

function svar_cutplane(
        X, # array of dimension N*T*D
        Y; # array of dimension N*T
        edges::Dict{Int, Vector{Int}}=Dict(
            t => t==1 ? (size(X,2)==1 ? Int[] : [t+1]) : (t==size(X,2) ? [t-1] : [t-1,t+1]) for t=1:size(X,2)
        ), # defaults to temporally varying model
        z0=zeros(size(X,2),size(X,3)),
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
        fullinverse::Bool=false,
        findif::Bool=false,
        verbose::Int=0 # 1 for solver output, 2 for timing inner problem
    )
    
    if @isdefined DBG
        s = "Mem before cutplane: $(memuse())"
        debug(DBG, :memory, s)
    end
    
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
    lnr.lnr_params[:fullinverse] = fullinverse
    lnr.lnr_params[:findif] = findif

    N,T,D = size(X)
    @assert sparsity<=global_sparsity "Global sparsity cannot be less than sparsity per time step."
    
    if (@isdefined DBG) & (verbose>0)
        s = "Starting cutting planes: N,T,D,sparsity = $((N,T,D,sparsity))"
        debug(DBG, :progress, s)
    end
    
    t0 = time()
    
    # Random starting point
    if sum(z0) == 0
        ind_init = sample(1:D,sparsity,replace=false)
        z0[:,ind_init] .= 1
    end
    z0 = convert.(Int, round.(z0))
    lnr.lnr_stats[:t_constant_computation] = @elapsed M, m, offset = compute_M(X,Y,edges,lambda_reg,lambda_svar,verbose=verbose)
    t_first_cut = @elapsed beta0,obj0,grad0 = inner_problem(M,m,offset,edges,z0,lambda_reg,lambda_svar,verbose=verbose)

    # Create model
    mio = (solver == :Gurobi) ? Model(Gurobi.Optimizer) : Model(CPLEX.Optimizer)
    set_optimizer_attribute(mio, (solver == :Gurobi) ? "TimeLimit" : "CPX_PARAM_TILIM", time_limit)
    set_optimizer_attribute(mio, (solver == :Gurobi) ? "OutputFlag" : "CPX_PARAM_SCRIND", 1*verbose)
    set_optimizer_attribute(mio, (solver == :Gurobi) ? "MIPGap" : "CPX_PARAM_EPGAP", mip_gap)
    set_optimizer_attribute(mio, (solver == :Gurobi) ? "Threads" : "CPXPARAM_Threads", 1)
    if (solver == :Gurobi) set_optimizer_attribute(mio, "NodefileStart", nodefilestart) end  

    # Variables
    @variable(mio, z[t=1:T,d=1:D], Bin, start=z0[t,d])
    @variable(mio, z_overall[d=1:D], Bin)
    if T>1
        @variable(mio, z_diff[t=1:T,t_adj in edges[t],d=1:D; t<t_adj], Bin)
    end
    @variable(mio, obj>=0)

    @objective(mio,Min,obj)

    # Sparsity constraints
    @constraint(mio, [t=1:T], sum(z[t,:]) <= sparsity)
    @constraint(mio, [t=1:T,d=1:D], z[t,d] <= z_overall[d])
    @constraint(mio, sum(z_overall) <= global_sparsity)
    if T>1
        @constraint(mio, [t=1:T,t_adj in edges[t],d=1:D; t<t_adj], z_diff[t,t_adj,d] >= z[t,d]-z[t_adj,d])
        @constraint(mio, [t=1:T,t_adj in edges[t],d=1:D; t<t_adj], z_diff[t,t_adj,d] >= -z[t,d]+z[t_adj,d])
        @constraint(mio, sum(z_diff) <= sparsely_varying)
    end

    # Add first cut
    cut_count, t_cut = 1, t_first_cut
    @constraint(mio, obj >= obj0 + sum(grad0 .* (z.-z0)))

    # Outer Approximation    
    function outer_approximation(cb_data)
        z_t = zeros(T,D)
        for t=1:T
            for d=1:D
                z_t[t,d] = callback_value(cb_data, z[t,d])
            end
        end
        t_new_cut = @elapsed beta_t,obj_t,grad_t = inner_problem(M,m,offset,edges,z_t,lambda_reg,lambda_svar,verbose=verbose)
        con = @build_constraint(obj >= obj_t + sum(grad_t .* (z.-z_t)) )
        MOI.submit(mio, MOI.LazyConstraint(cb_data), con)
        cut_count += 1
        t_cut += t_new_cut
    end
    MOI.set(mio, MOI.LazyConstraintCallback(), outer_approximation)
    
    mem = @allocated optimize!(mio)
    if @isdefined DBG
        s = "Mem during solver: $(mem/1024^2)"
        debug(DBG, :memory, s)
    end
    
    if has_values(mio)
        z_opt = JuMP.value.(z)
        if decrease_final_regularizer
            lambda_reg = sqrt(lambda_reg)
        end
        beta_opt,obj_opt,grad_opt = inner_problem(M,m,offset,edges,z_opt,lambda_reg,lambda_svar,verbose=verbose) 

        t_total = time()-t0

        lnr.beta, lnr.z, lnr.t = beta_opt, z_opt, t_total

        lnr.lnr_stats[:status] = termination_status(mio)
        lnr.lnr_stats[:t_solver] = JuMP.solve_time(mio)
        lnr.lnr_stats[:gap] = 1 - JuMP.objective_bound(mio) /  abs(JuMP.objective_value(mio)) 
        lnr.lnr_stats[:obj] = obj_opt
        lnr.lnr_stats[:grad] = grad_opt
        lnr.lnr_stats[:cut_count] = cut_count
        lnr.lnr_stats[:t_cut_total] = t_cut
        lnr.lnr_stats[:t_cut_avg] = t_cut/cut_count
    else
        t_total = time()-t0
        lnr.beta, lnr.z, lnr.t = zeros(T,D), z_opt(T,D), t_total
    end
    
    mio = nothing
    GC.gc()
    
    if @isdefined DBG
        s = "Mem after cutplane: $(memuse())"
        debug(DBG, :memory, s)
    end
    
    return lnr
    
end


#############################
# Inner problem - closed form
#############################

function compute_M(X,
                   Y,
                   edges,
                   lambda_reg,
                   lambda_svar;
                   verbose::Int=0)
    
    N,T,D = size(X)
    
    if @isdefined DBG
        s = "Mem before M computation: $(memuse())"
        debug(DBG, :memory, s)
    end
    
    # Create M and m
    M = spzeros(T*D,T*D)
    m = zeros(T*D)
    for t=1:T, d=1:D
        
        if @isdefined DBG
            if d==1
                s = "\t Mem during M, t=$t cutplane: $(memuse())"
                debug(DBG, :memory, s)
            end
        end
        
        # Current row of matrix M
        i = ind_1d(t,d,T,D)
        # Coefs corresp to squared error on variable i=(t,d)
        M[i,i] += sum(X[n,t,d]^2 for n=1:N)
        # Coefs corresp to objective on other variables i'=(t,d')
        for d2=(d+1):D
            j = ind_1d(t,d2,T,D)
            val = sum(X[n,t,d]*X[n,t,d2] for n=1:N)
            M[i,j] += val
            M[j,i] += val
        end
        # Coefs corresp to svar penalty with respect to adjacent vertices
        if T>1
            for t_adj in edges[t]
                M[i,i] += lambda_svar
                i_adj = ind_1d(t_adj,d,T,D)
                M[i,i_adj] -= lambda_svar
            end
        end
        # m vector entry
        m[i] += sum(Y[n,t]*X[n,t,d] for n=1:N)
    end
    
    offset = sum(Y.^2)
    
    if @isdefined DBG
        s = "Mem after M computation: $(memuse())"
        debug(DBG, :memory, s)
    end
    
    return M, m, offset
end

function inner_problem(M,
                       m,
                       offset,
                       edges,
                       z_2d,
                       lambda_reg::Real,
                       lambda_svar::Real;
                       compute_gradient::Bool=true,
                       fullinverse::Bool=false,
                       verbose::Int=0)
    
    if (@isdefined DBG) & (verbose>1)
        s = "\t - Solving inner problem:"
        debug(DBG, :progress, s)
    end
    
    T,D = size(z_2d)
    
    # Create Z matrix
    z = convert_1d(z_2d)
    Z = Diagonal(z)

    ##############
    # Compute beta
    ##############
    t0 = time()

    t_inv = time()
    
    if fullinverse
    
        K = inv(lambda_reg.*I+Z*M)
        
    else
    
        # Efficient implementation: solving linear system for beta_opt exploiting block structure of matrices
        vars = diag(Z).==1
        novars = diag(Z).==0    
        K = zeros(T*D,T*D)
        dense_submatrix = Matrix(M[vars,vars])
        small_inv = inv(lambda_reg.*I+dense_submatrix)
        K[vars,vars] = small_inv
        K[vars,novars] = - small_inv * M[vars,novars]./lambda_reg
        #  This is not actually the true K yet, the true K has the following formula: K = K + I - Z
        K[diagind(K)[novars]] .+= 1/ lambda_reg
        
    end
    
    Δt_inv = time() - t_inv
    
    beta = K[:,vars]*m[vars]
    beta[abs.(beta) .< 1e-4] .= 0
    beta_2d = convert_2d(beta,T,D)
    
    Δt = time() - t0
    if (@isdefined DBG) & (verbose>1)
        s = "\t - Compute beta: $Δt, t_inv: $Δt_inv"
        debug(DBG, :progress, s)
    end
    
    ###################
    # Compute objective
    ###################
    t0 = time()

    obj = -m[vars]'*beta[vars] + offset
    
    Δt = time() - t0   
    
    if (@isdefined DBG) & (verbose>1)
        s = "Compute obj: $Δt"
        debug(DBG, :progress, s)
    end
    
    ##################
    # Compute gradient
    ##################
    
    grad = nothing
    if compute_gradient
        
        t0 = time()

        grad = zeros(T,D)
        tmp0 = K[vars,vars]*m[vars] # O(T^2*K^2)

        for d_t=1:T, d_d=1:D

            d_i = ind_1d(d_t,d_d,T,D)

            elem1 = sum(M[d_i,vars].*tmp0)
            if z[d_i] == 1
                tmp1 = K[vars,d_i].*elem1 # O(T*K)
                tmp1 = sum(m[vars].*tmp1) # O(T*K)
                tmp2 = K[vars,d_i].*m[d_i] # O(T*K)
                tmp2 = sum(m[vars].*tmp2)        
            else
                tmp1 = K[vars,d_i].*elem1 # O(T*K)
                tmp1 = sum(m[vars].*tmp1)+m[d_i]*elem1/lambda_reg # O(T*K)
                tmp2 = K[vars,d_i].*m[d_i] # O(T*K)
                tmp2 = sum(m[vars].*tmp2)+m[d_i]*m[d_i]/lambda_reg # O(T*K)   
            end

            grad[d_t,d_d] = 0.5*(tmp1-tmp2)
        end

        Δt = time() - t0
        
        if (@isdefined DBG) & (verbose>1)
            s = "\t - Compute grad: $Δt"
            debug(DBG, :progress, s)
        end
        
    end
    
    return beta_2d,obj,grad
end

#################
# Utils: indexing
#################

function ind_1d(t,d,T,D)
    return d + (t-1) * D
end

function ind_2d(i,T,D)
    d = (i-1)%D+1
    t = convert(Int,floor((i-1)/D))+1
    return (t,d)
end

function convert_1d(x_2d)
    T,D = size(x_2d)
    x_1d = zeros(T*D)
    for t=1:T, d=1:D
        x_1d[ind_1d(t,d,T,D)] = x_2d[t,d]
    end
    return x_1d
end

function convert_2d(x_1d,T,D)
    x_2d = zeros(T,D)
    for i=1:T*D
        t,d = ind_2d(i,T,D)
        x_2d[t,d] = x_1d[i]
    end
    return x_2d
end