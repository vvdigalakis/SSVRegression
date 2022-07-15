using Parameters, Statistics

@with_kw mutable struct SvarLearner
    beta=nothing
    z=nothing
    t=0.
    lnr_params=Dict() # dictionary: params used for train
    lnr_stats=Dict() # dictionary: stats
end


###
# SOLUTION EVALUATION FUNCTIONS
###

function standardize_data(X_train, Y_train, X_valid, Y_valid, X_test, Y_test)
    
    N,T,D = size(X_train)
    
    for t=1:T
        for d=1:D
            μ, σ = mean(X_train[:,t,d]), std(X_train[:,t,d])
            σ = σ < 1e-3 ? 1. : σ
            X_train[:,t,d] .= (X_train[:,t,d] .- μ) ./ σ
            X_valid[:,t,d] .= (X_valid[:,t,d] .- μ) ./ σ
            X_test[:,t,d] .= (X_test[:,t,d] .- μ) ./ σ
        end
        μ, σ = mean(Y_train[:,t]), std(Y_train[:,t])
        σ = σ < 1e-3 ? 1. : σ
        Y_train[:,t] .= (Y_train[:,t] .- μ) ./ σ
        Y_valid[:,t] .= (Y_valid[:,t] .- μ) ./ σ
        Y_test[:,t] .= (Y_test[:,t] .- μ) ./ σ
    end
        
    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test
    
end


###
# SOLUTION EVALUATION FUNCTIONS
###   

function eval_sparsity(z, edges)
    return Dict(
        :local_sparsity => eval_local_sparsity(z),
        :global_sparsity => eval_global_sparsity(z),
        :changes_in_support => eval_sparsely_varying(z, edges)
    )
end    

function eval_local_sparsity(z)
    return maximum(sum(z .> 0, dims=2))
end

function check_local_sparsity(z, sparsity)
    return eval_local_sparsity(z) <= sparsity
end

function eval_global_sparsity(z)
    return sum(sum(z .> 0, dims=1) .> 0)
end

function check_global_sparsity(z, global_sparsity)
    return eval_global_sparsity(z) <= global_sparsity
end

function eval_sparsely_varying(z, edges)
    T = size(z,1)
    changes = 0
    if T>1
        for t=1:T
            for t_adj in edges[t]
                changes += 0.5*sum(z[t,:].!=z[t_adj,:])
            end
        end
    end
    return changes
end

function check_sparsely_varying(z, edges, sparsely_varying)
    return eval_sparsely_varying(z, edges) <= sparsely_varying
end

function eval_differences_in_support(z_true,z_est)
    return sum(abs.(z_true .- z_est))
end

function eval_support_recovery(z_true,z_est)
    nnz_actual,nnz_detected,nnz_false = 0,0,0
    for i=1:length(z_true)
        if z_true[i] > .5
            nnz_actual += 1
            if z_est[i] > .5
                nnz_detected += 1
            end
        end
        if (z_true[i] < .5) & (z_est[i] > .5)
            nnz_false += 1
        end
    end
    return (nnz_detected/nnz_actual*100),(nnz_false/(nnz_false+nnz_actual)*100)
end

function eval_mae_in_coefs(beta_true,beta_est)
    T,D = size(beta_true)
    return sum(abs.(beta_true .- beta_est))/T/D
end

function eval_r2(X,Y,beta)
    N,T,D = size(X)
    pred = predict(X,beta)
    SSres = sum((Y.-pred).^2)
    SStot = sum((Y.-mean(Y)).^2)
    return 1-SSres/SStot
end

function predict(X,beta)
    N,T,D = size(X)
    pred = zeros(N,T)
    for n=1:N, t=1:T
        pred[n,t] = sum(X[n,t,:].*beta[t,:])
    end
    return pred
end

        
###
# DEBUG FUNCTIONS
###
        
function memuse()
    pid = getpid()
    return round(Int,parse(Int,read(`ps -p $pid -o rss=`,String))/1024)
end

# debug function: write string to file
function debug(DBG_out, DBG_type, s::String)
#     if !(@isdefined DBG_list)
#         DBG_list = [:memory, :progress, :solver]
#     end
    if DBG_type in DBG_list
        if DBG_type == :memory
            s = "\n\n--\n - MEM:\n"*s*"\n--\n"
        elseif DBG_type == :progress
            s = "\n\n--\n - PRG:\n"*s*"\n--\n"
        elseif DBG_type == :solver
            s = "\n\n--\n - SOLVER:\n"*s*"\n--\n"
        end
        if DBG_out != ""
            if DBG_out == stdout
                println(s)
            else
                open(DBG_out,"a") do io
                    write(io,s)
                end
            end
        end
    end
end