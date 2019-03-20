"""
Applies CausalKinetiX framework to rank variables and models according to their stability.

Parameters
----------

D : data matrix (2d-array). Should have dimension n x (L*d), where n is the number of repetitions (over all experiments), L is the number of time points and d is the number of predictor variables.
times : 1d-array of length L specifying the time points at which data was observed.
env : integer 1d-array of length n encoding to which experiment each repetition belongs. be aware that variable index starts from 0.
target : integer specifing which variable is the target.
models : list of models. Each model is specified by a list of lists specifiying the variables included in the interactions of each term. for example, [[0],[1,2]] stands for a model whose predictor variables are x0 and x1*x2
max_preds : (default FALSE) if TRUE also models with lower terms included.
expsize : (default 2) the expected number of terms.
interactions : (default FALSE) specifies whether to include interactions in the models.
products : (default FALSE) specifies whether to include products in the models.
include_vars : (default None) specifies variables that should be included in each model. use -1 for using original variables(e.g. include_var=-1 returns same result as inculude_var=None). 
maineffect_models: (default FALSE) main-effect models or exhaustive models. 
screening : (default None) specifies how many variables to screen down.
K : (default None) cutoff paramter used in variable ranking.        

Returns
-------
dict object with following keys and values
"models" : list of the individually scored models.  
"model_scores" : 1d-array containing the score for each model.
"variable_scores" : 1d-array containing the score of each variable.
"ranking" : 1d-array specifying the ranking of each variable.

Examples
--------
## Generate data from Maillard reaction
>>> simulation_obj = generate_data_maillard(env=np.array(list(range(5))*3),
                                        L=15,
                                        target=0)
>>> D = simulation_obj["simulated_data"]
>>> time = simulation_obj["time"]
>>> env = simulation_obj["env"]
>>> target = simulation_obj["target"]
>>> # plot the observation of target variable in environment 0
>>> plt.plot(time, D[0,-len(time):], '-',c="black")
>>> plt.plot(time, D[0,-len(time):], 'o',c="red")
>>> plt.legend(["observations"])
>>> # output of following is random
>>> CausalKinetiX(D = D, times = time, env=env, target = target)
{'models': [[[0], [1], [2]],
  [[0], [1], [3]],
  [[0], [1], [4]],
  [[0], [2], [3]],
  [[0], [2], [4]],
  [[0], [3], [4]],
  [[1], [2], [3]],
  [[1], [2], [4]],
  [[1], [3], [4]],
  [[2], [3], [4]]],
 'model_scores': array([0.0048267 , 0.00468692, 0.00464499, 1.10973326, 0.23748128,
        0.16553816, 0.34931807, 0.17851886, 0.25998287, 0.19644722]),
 'variable_scores': array([0.        , 0.        , 0.66666667, 0.66666667, 0.66666667]),
 'ranking': array([0, 1, 2, 3, 4])}

Notes
-----
The function CausalKinetiX_modelranking can be used if the variable ranking is not required.
For further details see the following references.
Pfister, N., S. Bauer, J. Peters (2018).
Identifying Causal Structure in Large-Scale Kinetic Systems
(https://arxiv.org/pdf/1810.11776.pdf)
"""


import numpy as np
import scipy.stats
from .CausalKinetiX_modelranking import CausalKinetiX_modelranking
from .utils import construct_models


def CausalKinetiX(D,
                  times,
                  env,
                  target,
                  models=None,
                  max_preds=False,
                  expsize=2,
                  interactions=False,
                  products=False,
                  include_vars=None,
                  maineffect_models=True,
                  rm_target=False,
                  screening=None,
                  K=None,
                  regression_class=None ):
    
    # for random forest regression remove interactions
    if regression_class == "random_forest":
        interactions = False
        interactions_Y = False
        include_intercept <- False
    if regression_class == None:
        regression_class = 'OLS'
    
    # read out variables
    n = D.shape[0]
    L = len(times)
    d = D.shape[1]//L  

    # Check whether a list of models was specified else generate models
    if models==None:
        constructed_mods = construct_models(D, L, d, n, target, times,
                                         maineffect_models,
                                         screening,
                                         interactions,
                                         products,
                                         include_vars,
                                         max_preds,
                                         expsize,
                                         env)
        models = constructed_mods["models"]
        if K==None:
            K = constructed_mods["num_terms"]-expsize

    # check whether parameter K was specified
    if K==None:
        print("K was not specified and the default does not make sense for arbitrary lists of models. It was set to 1, but this can invalidate the variable ranking.")
        K = 1

    ###
    # Compute model scores
    ###
    
    
    model_scores = CausalKinetiX_modelranking(
        D, 
        times, 
        env, 
        target, 
        models, 
        include_vars = include_vars,
        regression_class = regression_class,
        #in R implementation, "mean" is used instead. 
        score_type = "mean2"
    )

    ###
    # Rank variables
    ###

    Mlen = len(models)
    Mjlen = np.array([
                    sum(sum([[([x] == term) for term in mod] for mod in models],[]))
                for x in range(d)])
    # compute p-values based on hypergeometric distribution
    best_mods = np.array([[]]+models, dtype=np.object)[1:][model_scores.argsort()][range(K)]
    counts = np.array([
                        sum(sum([[[x] == term for term in mod] for mod in best_mods],[]))
                    for x in range(d)])
    var_scores = (1./K) * counts
    var_pvals = np.array([
                        scipy.stats.hypergeom.sf(counts[j], Mlen, Mjlen[j], K) 
                    for j in range(d)])

    var_pvals[Mjlen==0] = np.Infinity

    idx = var_pvals.argsort()
    ranking = np.arange(d)[idx]
    scores = var_pvals[idx]

       
    # output results
    return({"models":models,
          "model_scores":model_scores,
          "variable_scores":scores,
          "ranking":ranking})