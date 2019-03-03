import numpy as np
import scipy.stats
from CausalKinetiX_modelranking import CausalKinetiX_modelranking
from utils import construct_models


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
        warning("K was not specified and the default does not make sense for arbitrary lists of models. It was set to 1, but this can invalidate the variable ranking.")
        K = 1

    ###
    # Compute model scores
    ###

    model_scores = CausalKinetiX_modelranking(D, times, env, target, models, include_vars=include_vars)

    ###
    # Rank variables
    ###

    Mlen = len(models)
    Mjlen = np.array([
                    sum(sum([[(x in term) for term in mod] for mod in models],[]))
                for x in range(d)])
    # compute p-values based on hypergeometric distribution
    best_mods = np.array([[]]+models, dtype=np.object)[1:][model_scores.argsort()][range(K)]
    counts = np.array([
                        sum(sum([[(x in term) for term in mod] for mod in best_mods],[]))
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