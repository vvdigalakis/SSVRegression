using Parameters, DataFrames

include("svar_cutplane.jl")
include("svar_heuristic.jl")
include("regression_sum_of_norms.jl")
include("sparse_regression.jl")

@with_kw mutable struct GridSearch
    lnr_refit=nothing
    grid_results=nothing
    generated_warm_start_dict=nothing
end

function holdout_grid_search( 
        X, # array of dimension N*T*D
        Y, # array of dimension N*T
        method,
        search_params,
        const_params; 
        X_valid=nothing,
        Y_valid=nothing,
        split_at::Real=0.7,
        param_decrease_factor=2,
        warm_start_dict=nothing, # dictionary mapping search params to warm starts -- not supported yet
        generate_warm_start_dict=false,
        verbose::Int=0
    )
    
    N,T,D = size(X)
    
    # Data split
    if (X_valid == nothing) | (Y_valid == nothing)
        N_train = convert(Int, floor(split_at*N))
        train_indices = sample(collect(1:N), N_train, replace=false)
        X_train = X[train_indices,:,:]
        Y_train = Y[train_indices,:]
        valid_indices = setdiff(collect(1:N),train_indices)
        X_valid = X[valid_indices,:,:]
        Y_valid = Y[valid_indices,:]
    else 
        N_train = N
        X_train = X[:,:,:]
        Y_train = Y[:,:]
        X = vcat(X_train,X_valid)
        Y = vcat(Y_train,Y_valid)
    end
    
    # Put search params in list
    tmp = Dict{Symbol,Any}()
    for (k,v) in search_params
        tmp[k] = v
    end
    search_params = tmp
    for (p_name,p_val) in search_params
        # If param is provided as Int, start with default value and decrease
        if isa(p_val, Int)
            # Get param default
            p0, param_type = get_param_defaults(p_name,N,D)
            p_list = []
            for i=1:p_val
                new_val = p0/param_decrease_factor^(i-1)
                if param_type == :int
                    new_val = convert(Int, ceil(new_val))
                end
                append!(p_list, new_val)
            end
            search_params[p_name] = sort(unique(p_list))
        end
    end
    
    search_param_names = collect(keys(search_params))
    search_param_combinations = [
        Dict(search_param_names[i]=>p[i] for i in 1:length(search_param_names))
        for p in Iterators.product([search_params[i] for i in keys(search_params)
        ]...)
    ]

    # Validation array
    cv = DataFrame()
    insertcols!(cv, (search_param_names .=> Ref([]))...)
    insertcols!(cv, ([:score, :lnr] .=> Ref([]))...)
    
    if length(search_param_combinations) > 1

        # Run validation
        for (comb_num, search_param_comb) in enumerate(search_param_combinations)

            if @isdefined DBG
                s = "CV $comb_num of $(length(search_param_combinations)) params: $(search_param_comb)"
                debug(DBG, :progress, s)
            end

            if @isdefined DBG
                s = "Mem before method: $(memuse())"
                debug(DBG, :memory, s)
            end

            # Fit learner for current param combination
            lnr = method(X_train, Y_train; search_param_comb..., const_params...)        

            if @isdefined DBG
                s = "Mem after method: $(memuse())"
                debug(DBG, :memory, s)
            end

            # Evaluate
            score = eval_r2(X_valid,Y_valid,lnr.beta)
            new_row = Dict()
            for (k,v) in search_param_comb
                new_row[k] = v
            end
            new_row[:score], new_row[:lnr] = score, lnr
            cv = vcat(cv, DataFrame(new_row))
        end
        
    else
        new_row = Dict()
        for (k,v) in search_param_combinations[1]
            new_row[k] = v
        end
        new_row[:score], new_row[:lnr] = -1, nothing
        cv = vcat(cv, DataFrame(new_row))
        
    end

    # Find best param comb and refit
    best = argmax(cv[!,:score])
    best_search_param_comb = cv[best,search_param_names]
    lnr = method(X, Y; best_search_param_comb..., const_params...) 
    
    return GridSearch(lnr_refit=lnr, grid_results=cv)
end

function get_param_defaults(param_name,N,D)
    if param_name == :lambda_reg
        return N, :real
    elseif param_name == :lambda_svar
        return sqrt(N), :real
    elseif param_name == :sparsity
        # take 10% of features by default
        return min(50, convert(Int, ceil(0.2*D))), :int
    elseif param_name == :global_sparsity
        # take 15% of features by default
        return min(70, convert(Int, ceil(0.3*D))), :int
    elseif param_name == :sparsely_varying
        # 1 change in support counts for 2 in terms of the sparsely varying parameter
        return min(40, 2*convert(Int, ceil(0.1*D))), :int
    end
end