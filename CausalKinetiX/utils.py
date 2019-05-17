####
# Small helper functions that are not exported
###

import numpy as np
from glmnet import ElasticNet
from itertools import combinations


## Lasso based on deltaY vs intX
def ode_integratedlasso_rank_vars(D,
                                  times,
                                  target,
                                  env=None,
                                  silent=True,
                                  interactions=True,
                                  rm_target=True):

    L = len(times)
    d = D.shape[1]//L
    n = D.shape[0]

    Xint = np.zeros([(L-1)*n, d])
    deltaY = np.zeros([(L-1)*n])
    for i in range(n):
        deltaY[(i)*(L-1):(i+1)*(L-1)] = np.diff(D[i, target*L:(target+1)*L])
    for j in range(d):
        for i in range(n):
            tmp = D[i, j*L:(j+1)*L]
            Xint[i*(L-1):(i+1)*(L-1), j] = (tmp[:(L-1)]+tmp[1:])/2*np.diff(times)

    # remove NAs
    na_ind = np.logical_or(np.isnan(deltaY), (np.isnan(Xint) > 0).sum(axis=1))
    deltaY = deltaY[~na_ind]
    Xint = Xint[~na_ind, ]

    # Perform lasso
    if interactions:
        dC2 = d*(d-1)//2 # combination
        var_names = np.zeros([d+dC2+d], dtype=np.object)
        var_names[:d] = np.array([[]]+[[i] for i in range(d)], dtype=np.object)[1:]
        var_names[d:] = np.array([[]]+sum(([[i, j] for j in range(i+1)] for i in range(d)), []), dtype=np.object)[1:]
        Xint_interactions = np.zeros([n*(L-1), len(var_names)])
        Xint_interactions[:, :d] = Xint
        for i in range(d, len(var_names)):
            Xint_interactions[:, i] = Xint[:, var_names[i][0]] * Xint[:, var_names[i][1]]
        fit = ElasticNet().fit(Xint_interactions, deltaY)
        sel_matrix = (np.abs(fit.coef_path_) > 1e-7)
        first_entrance = sel_matrix.max(axis=1)
        # find all rows without ones and set first entrance to Inf
        first_entrance[sel_matrix.sum(axis=1) == 0] = np.infty
        ranking = first_entrance.argsort()
        ranking = var_names[ranking]
    else:
        fit = ElasticNet().fit(Xint, deltaY)
        sel_matrix = fit.coef_path_ != 0
        first_entrance = sel_matrix.max(axis=1)
        # find all rows without ones and set first entrance to Inf
        first_entrance[sel_matrix.sum(axis=1) == 0] = np.infty
        ranking = first_entrance.argsort()
    if rm_target:
        ranking = ranking[ranking != target]

    return({'ranking': ranking, 'coef': fit.coef_})


def construct_models(D, L, d, n, target, times,
                     maineffects_models, screening,
                     interactions, products, include_vars,
                     max_preds, expsize, env=None):

    ## Main-Effect and Full-Effect models depends on maineffects_models
    # R implementation seems to be wrong
    # if not maineffects_models==True:
    if maineffects_models:
        # construct variable vector
        if not isinstance(include_vars, type(None)):
            include_vars = np.array(include_vars)
            # for utility
            unmatch = lambda pattern, matched: \
                np.arange(len(matched))[np.equal(
                    np.array(pattern).reshape([-1, 1]),
                    np.array(matched).reshape([1, -1])
                ).sum(axis=0) == 0]
            vv = unmatch(
                include_vars[include_vars >= 0],
                np.arange(d)
            )
        else:
            vv = np.arange(d)

        ## Decide which terms to keep depending on screening, interactions and products
        if type(screening) == int:
            tmp = extend_Dmat(D, L, d, n, products, interactions, include_vars)
            Dfull = tmp["Dnew"]
            ordering = tmp["ordering"]
            res = ode_integratedlasso_rank_vars(Dfull,
                                                times,
                                                target,
                                                env=env,
                                                interactions=False,
                                                rm_target=False)["ranking"]
            keep_terms = ordering[res[range(screening)]]
            print(keep_terms)
            num_terms = screening
        else:
            keep_terms = [[v] for v in vv]
            tmp_terms = []
            # add interactions
            if interactions:
                tmp_terms += [list(tupl) for tupl in combinations(list(vv), 2)]
            # add products
            if products:
                keep_terms += [[i, i] for i in vv]
            # include_vars
            if not isinstance(include_vars, type(None)):
                keep_terms_new = []
                for i in range(len(keep_terms)):
                    for var in include_vars:
                        if var < 0:
                            keep_terms_new.append(keep_terms[i])
                        else:
                            tmp_term = [var] + keep_terms[i]
                            tmp_term.sort()
                            keep_terms_new.append(tmp_term)
                keep_terms = keep_terms_new + [[term] for term in include_vars[~include_vars<0]]
            num_terms = len(keep_terms)

        ## Construct models
        if max_preds:
            models = []
            for k in range(1, expsize+1):
                models += [list(tupl) for tupl in combinations(keep_terms, k)]
        else:
            models = [list(tupl) for tupl in combinations(keep_terms, expsize+1)]
    else:
        if not isinstance(include_vars, type(None)):
            print("include_vars is not defined for maineffects_models==FALSE")
        ## Construct models
        if max_preds:
            models = []
            for k in range(1, expsize+1):
                models += [list(tupl) for tupl in combinations(np.arange(d), k)]
        else:
            models = [list(tupl) for tupl in combinations(np.arange(d), expsize+1)]
        # add interactions and products
        for i in range(len(models)):
            if len(models[i]) == 1:
                models[i] = [models[i]]
            if len(models[i]) > 1:
                tmp_model = []
                if interactions:
                    tmp_model += [list(tupl) for tupl in combinations(models[i], 2)]
                if products:
                    tmp_model += [[i,i] for i in models[i]]
                models[i] = [[term] for term in models[i]] + tmp_model
        num_terms = d

    # return output
    result = {"models": models,
              "num_terms": num_terms}
    return result




def extend_Dmat(D, L, d, n,
                products,
                interactions,
                include_vars):

    ### include_vars is supported in different way from R implementation now
    ### include_var[i] < 0 indicates the original variable (not include_var[i] == 0)
    assert(type(L) == type(d) == type(n) == int)
    if not isinstance(include_vars, type(None)): # different from implementation in R
        # construct variable vector
        include_vars = np.array(include_vars)
        # for utility
        unmatch = lambda pattern, matched: \
            np.arange(len(matched))[np.equal(
                np.array(pattern).reshape([-1, 1]),
                np.array(matched).reshape([1, -1])
            ).sum(axis=0) == 0]
        vv = unmatch(
            include_vars[include_vars < 0],
            np.arange(d)
        )
        vv_ind = np.array(sum([list(range(var*L, (var+1)*L)) for var in vv], []))
    else:
        vv = np.arange(d)
        vv_ind = np.arange(L*d)
    dnew = len(vv)

    # initialize
    dC2 = dnew*(dnew-1)//2 # combination
    num_vars = dnew + products*dnew + interactions*(dC2+d)
    Dnew = np.zeros([n, L*num_vars])
    Dnew[:, range(dnew*L)] = D[:, vv_ind]
    ordering = np.zeros([num_vars], dtype=np.object)
    for i in range(dnew):
        ordering[i] = [vv[i]]
    count = dnew

    # add interactions
    if interactions:
        for i in range(dnew):
            indi = np.arange(vv[i]*L, (vv[i]+1)*L)
            for j in range(i, dnew):
                ordering[count] = [vv[i], vv[j]]
                indj = np.arange(vv[j]*L, (vv[j]+1)*L)
                Dnew[:, np.arange(count*L, (count+1)*L)] = D[:, indi]*D[:, indj]
                count = count+1

    # add products
    if(products):
        for i in range(dnew):
            indi = np.arange(vv[i]*L, (vv[i]+1)*L)
            ordering[count] = [vv[i], vv[i]]
            Dnew[:, count*L:(count+1)*L] = D[:, indi]**2
            count = count + 1

    # include variables to every term
    if not isinstance(include_vars, type(None)):
        Dfinal = np.zeros([Dnew.shape[0], len(include_vars)*Dnew.shape[1]+sum(~include_vars < 0)*L])
        ordering_final = np.zeros([len(include_vars)*len(ordering)+sum(~include_vars < 0)], dtype=np.object)
        count = 0
        for var in include_vars[~include_vars < 0]:
            Dfinal[:, count*L:(count+1)*L] = D[:, var*L:(var+1)*L]
            ordering_final[count] = [var]
            count = count + 1
        for j in range(len(ordering)):
            for var in include_vars:
                if var < 0:
                    ordering_final[count] = ordering[j]
                    Dfinal[:, count*L:(count+1)*L] = Dnew[:, j*L:(j+1)*L]
                else:
                    tmp_order = np.append(ordering[j], var)
                    tmp_order.sort()
                    ordering_final[count] = list(tmp_order)
                    Dfinal[:, count*L:(count+1)*L] = D[:, var*L:(var+1)*L] * Dnew[:, j*L:(j+1)*L]
                count = count + 1
        ordering = ordering_final
        Dnew = Dfinal

    return({"Dnew": Dnew,
            "ordering": ordering})
