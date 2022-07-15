# SSVRegression

This repository contains contains code for the algorithms described in the paper **Slowly Varying Regression under Sparsity** (preprint available on arXiv).

The Slowly Varying Regression under Sparsity framework enables sparse regression models to slowly and sparsely change over time or space or any other dimension. 

The algorithms implemented here include:
- ```svar_cutplane```: cutting plane algorithm that exactly solves the Slowly Varying Regression under Sparsity mixed-integer optimization formulation. Key arguments include:
  - ```X```: data matrix, array of dimension NxTxD.
  - ```Y```: responses, array of dimension NxT.
  - ```edges``` (optional): similarity graph, dictionary mapping each vertex (Int) to its adjacent vertices (Int[]). Defaults to chain, i.e., temporally varying similarity graph.
  - ```z0``` (optional): support of warm start point, (binary) array of dimension TxD.
  - ```sparsity``` (optional): local sparsity hyperparameter, Int.
  - ```global_sparsity``` (optional): global sparsity hyperparameter, Int.
  - ```global_sparsity_relative``` (optional): ```global_sparsity```-```sparsity``` (if use this parameter, do not use ```global_sparsity```), Int.
  - ```sparsely_varying``` (optional): sparsely varying support hyperparameter (total number of changes in support between all pairs of adjacent vertices), Int.
  - ```lambda_reg``` (optional): regularization weight hyperparameter, Real. Defaults to N.
  - ```lambda_svar``` (optional): slow variation penalty hyperparameter, Real. Defaults to sqrt(N).
  - ```time_limit``` (optional): solver time limit, Real.

- ```svar_heuristic```: fast algorithm that heuristically solves the Slowly Varying Regression under Sparsity mixed-integer optimization formulation. Key arguments are as in ```svar_cutplane``` (excluding ```z0```).

- ```regression_sum_of_norms```: solves the surrogate slowly varying regression with sum-of-norms and (possibly) lasso regularization. Key arguments are as in ```svar_cutplane``` (excluding ```z0```, ```sparsity```, ```global_sparsity```, ```global_sparsity_relative```, ```sparsely_varying```), in addition to the following:
  - ```norm``` (optional): :l1 or :l2 norm used in sum-of-norms regularization term.
  - ```lasso``` (optional): whether to use lasso regularizer, Bool.

- ```sparse_regression```: cutting plane algorithm that exactly solves the standard sparse regression formulation (without variation). Key arguments are ```X```, ```Y```, ```sparsity```, ```lambda_reg```, ```time_limit``` (defined as in ```svar_cutplane```).

For an example, see the SSVRegression-demo.ipynb notebook.

#### Compatibility note: 
We implement all algorithms in Julia programming language (version 1.6) and using the JuMP.jl modeling language for mathematical optimization (version 0.21). We solve the optimization models using the Gurobi commercial solver (version 9.5).
