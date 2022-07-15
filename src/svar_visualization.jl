using GraphRecipes, Plots, ColorSchemes, LaTeXStrings

function create_coef_df(
    beta;
    feature_names=[],
    vertex_names=[],
    top_D=5
)

    T,D = size(beta)

    if length(feature_names) == 0
        feature_names = ["feature $d" for d=1:D]
    end
    if length(vertex_names) == 0
        vertex_names = ["vertex $t" for t=1:T]
    end

    coef_df = DataFrame([
        :feature => [],
        :mean => [],
        :std => []
    ])

    for t=1:T
        coef_df[!,vertex_names[t]] = []
    end

    beta_mean = mean(beta,dims=1)[1,:]
    beta_std = std(beta,dims=1)[1,:]
    sorted_ind = sortperm(abs.(beta_mean), rev=true)

    for d=1:min(D,top_D)
        coefs = [
            feature_names[sorted_ind[d]],
            beta_mean[sorted_ind[d]],
            beta_std[sorted_ind[d]]            
        ]
        for t=1:T
            append!(coefs,beta[t,sorted_ind[d]])
        end
        push!(coef_df,coefs)
    end

    return coef_df
end

function plot_feature_variation(
        coef_df,
        edges;
        selected_feature=:top,
        colorscheme=ColorSchemes.roma,
        method=:stress,
        fontsize=8,
        output_file="",
    )
    
    vertex_names = names(coef_df)[4:end]
    
    if selected_feature == :top
        selected_feature = coef_df.feature[1]
    end
    
    ###
    # Extract graph adjacency matrix
    ###
    T = length(vertex_names)
    A = zeros(T,T)
    for t1=1:T-1, t2=t1+1:T
        if t1 in edges[t2]
            A[t1,t2] = 1
            A[t2,t1] = 1
        end
    end
    
    ###
    # Extract scaled coefs
    ###
    betas = coef_df[coef_df.feature .== selected_feature, :]
    beta_v = Real[]
    for (i,vertex) in enumerate(vertex_names)
        append!(beta_v, betas[1, vertex])
    end
    # Zero mean coefs
    beta_mean = betas[1, :mean]
    beta_std = betas[1, :std]
    beta_v_scaled = beta_v .- beta_mean
    # Find coef range in form [-c,c]
    beta_range = max(maximum(beta_v), -minimum(beta_v))
    # Scale to [0,1] where 0.5 corresponds to the mean
    beta_v_scaled = 1 .* (beta_v_scaled .+ beta_range) ./ (2*beta_range) .- 0
    
    ###
    # Convert coefs to colors
    ###
    color_v = []
    for (i,vertex) in enumerate(vertex_names)
        x = get(colorscheme, beta_v_scaled[i])
        append!(color_v, [(x.r, x.g, x.b)])
    end
    color_v = [parse(Colorant, RGB(c[1],c[2],c[3])) for c in color_v]
    
    ###
    # Generate plot
    ###
    
    name_and_coef = ["$v:\n $(round(beta_v[i],digits=2))" for (i,v) in enumerate(vertex_names)]
    title = L"Variation\ of\ \beta_{%$(selected_feature)}\ (mean=%$(round(beta_mean,digits=2)), \ std=%$(round(beta_std,digits=2)))"
    
    g = graphplot(
        A,
        nodeshape=:circle,
        method=method,
        markercolor = color_v,
        names = name_and_coef,
        linecolor = :black,
        fontsize = fontsize,
        title = title,
    )
    
    if output_file != ""
        savefig(output_file)
    end
    
    return g    
end