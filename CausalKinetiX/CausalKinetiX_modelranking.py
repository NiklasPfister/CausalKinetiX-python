"""
Applies CausalKinetiX framework to rank variables and models according to their stability.

This function scores a specified list of models and does not include a variable ranking.

Parameters
----------

D : data matrix(2d-array). Should have dimension n x (L*d), where n is the number of repetitions (over all experiments), L is the number of time points and d is the number of predictor variables.
times : 1d-array of length L specifying the time points at which data was observed.
env : integer 1d-array of length n encoding to which experiment each repetition belongs.
target : integer specifing which variable is the target.
models : list of models. Each model is specified by a list of lists specifiying the variables included in the interactions of each term. for example, [[0],[1,2]] stands for a model whose predictor variables are x0 and x1*x2.
pen_degree : (default 2) specifies the penalization degree in the smoothing spline.
num.folds : (default 2) number of folds used in cross-validation of smoothing spline.
include_vars : (default None) specifies variables that should be included in each model. use -1 for using original variables(e.g. include_var=-1 returns same result as inculude_var=None).
include_intercept : (default FALSE) specifies whether to include a intercept in models.
average_reps : (default False) specifies whether to average repetitions in each environment.
smooth_X : (default False) specifies whether to smooth predictor observations before fitting.
smooth_Y : (default FALSE) specifies whether to smooth target observations before fitting.
regression_class : (default OLS) other options are signed.OLS, optim, random.forest.
sample_splitting : (default "loo") either leave-one-out (loo) or no splitting (none).
score_type : (default "mean") : specifies the type of score funtion to use.
integrated_model : (default TRUE) specifies whether to fit the integrated or the derived model.
splitting_env : (default None) an additonal environment 1d-array used for scoring.
weight_vec : (default rep(1, length(env)) a weight 1d-array used in the scoring.
set_initial : (default FALSE) specifies whether to fix the initial value.
silent : (default TRUE) turn of additional output.
show_plot (default FALSE) show diagnostic plots.

Returns
-------
scores : 1d-array with the same length as models containing the stability scores. score_type="mean2" is used in Python implementation while "mean1" is used in R version. this causes the difference in scores between the two versions.

Examples
--------
## Generate data from tergetmodel reaction
>>> simulation_obj = generate_data_maillard(env=np.array(list(range(5))*3),
                                        L=15,
                                        target=0)
>>> D = simulation_obj["simulated_data"]
>>> time = simulation_obj["time"]
>>> env = simulation_obj["env"]
>>> target = simulation_obj["target"]
>>> true_model = simulation_obj["true_model"]
>>> models = [true_model]
>>> # plot the observation of target variable in environment 0
>>> plt.plot(time, D[0,-len(time):], '-',c="black")
>>> plt.plot(time, D[0,-len(time):], 'o',c="red")
>>> plt.legend(["observations"])
>>> # output of following is random
>>> CausalKinetiX_modelranking(D, time, env, target, models,
                           include_vars=None, show_plot=True,
                           integrated_model=False, score_type="mean2")
array([0.00598046])

Notes
-----
The function CausalKinetiX is a wrapper for this function that also computes the variable ranking.
For further details see the following references.
Pfister, N., S. Bauer, J. Peters (2018).
Identifying Causal Structure in Large-Scale Kinetic Systems
(https://arxiv.org/pdf/1810.11776.pdf)
"""


import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from quadprog import solve_qp
from matplotlib import pyplot as plt
from copy import deepcopy
from .constrained_smoothspline import constrained_smoothspline
from .utils import extend_Dmat

def CausalKinetiX_modelranking(D,
                               times,
                               env,
                               target,
                               models,
                               pen_degree=2,
                               num_folds=2,
                               include_vars=None,
                               include_intercept=False,
                               average_reps=False,
                               smooth_X=False,
                               smooth_Y=False,
                               regression_class="OLS",
                               sample_splitting="loo",
                               score_type="mean",
                               integrated_model=True,
                               splitting_env=None,
                               weight_vec=None,
                               set_initial=False,
                               silent=True,
                               show_plot=False):

    ## Set default parameter
    if isinstance(weight_vec, type(None)):
        weight_vec = np.ones([len(env)])
    if splitting_env is None:
        splitting_env = deepcopy(env)

    ## Parameter consistency checks
    if(smooth_Y and (not isinstance(splitting_env, type(None)))):
        raise(Exception("If smooth.Y is TRUE, splitting.env needs to be NA."))
    assert(sum(type(model) == list for model in models))
    assert(type(env) == np.ndarray)
    assert(len(set(times)) == len(times))

    ############################
    #
    # initialize
    #
    ############################

    # read out parameters (as in paper)
    n = D.shape[0]
    L = len(times)
    d = D.shape[1]//L

    if type(pen_degree) == int:
        pen_degree = np.repeat(pen_degree, 2)

    # check whether to include products and interactions
    products = any(len(term) > len(set(term)) for term in sum(models, []))
    interactions = any(len(set(term)) > 1 for term in sum(models, []))

    # sort environments to increasing order
    splitting_env.sort()
    D = D[splitting_env, ]
    if smooth_Y:
        weight_vec = weight_vec[list(set(env))] #list(set(*)) is equivalent to order(unique(*))
    else:
        weight_vec = weight_vec[splitting_env]

    # construct DmatY
    target_ind = np.arange(target*L, (target+1)*L)
    DmatY = D[:, target_ind]

    # add interactions to D-matrix if interactions==TRUE
    if (interactions or products or (not include_vars is None)):
        if not include_vars is None:
            interactions = True
            products = True
        include_obj = extend_Dmat(D, L, d, n,
                                  products=products,
                                  interactions=interactions,
                                  include_vars=include_vars)
        D = include_obj["Dnew"]
        ordering = include_obj["ordering"]
        dtot = D.shape[1]//L

    else:
        dtot = d
        ordering = np.array([[i] for i in range(d)], dtype=np.object)

    ##################################################
    #
    # Step 1: Average Repetitions & Smooth predictors
    #
    #################################################

    ## Averaging
    
    ## use averaging on environments
    if average_reps:
        unique_env = list(set(env))
        D_new = np.zeros([len(unique_env),D.shape[1]])
        DmatY_new = np.zeros([len(unique_env),L])
        
        for i in range(len(unique_env)):
            D_new[i,:] = D[env == unique_env[i],:].mean(axis=1)
            D_new[i,:] = D[env == unique_env[i],:].mean(axis=1)
        
        splitting_env = list(set(splitting_env))
        env = unique_env
        n = len(env)

    ## Smoothing
    Dlist = np.zeros([n], np.object)
    if smooth_X==True:
        for i in range(n):
            Dlist[i] = D[i,].reshape([dtot, L]).T
            # smooth X-values
            for j in range(dtot):
                na_ind = np.isnan(Dlist[i][:,j])
                f_approx = interp1d(times[~na_ind], Dlist[i][~na_ind,j], kind="cubic")
                Dlist[i][:,j] = f_approx(times)

    ## without smoothing
    else:
        for i in range(n):
            Dlist[i] = D[i,:].reshape([dtot, L]).T

    ######################################
    #
    # Step 2: Fit reference model
    #
    #####################################

    if smooth_Y==True:
        # initialize variables
        unique_env = list(set(env))
        num_env = len(unique_env)
        Ylist = np.zeros([num_env], dtype=np.object)
        envtimes = np.zeros([num_env], dtype=np.object)
        dYlist = np.zeros([num_env], dtype=np.object)
        RSS_A = np.zeros([num_env])
        UpDown_A = np.zeros([num_env])
        RSS3_A = np.zeros([num_env])
        lambd = np.zeros([num_env])
        initial_values = np.zeros([num_env])
        times_new = np.array([0])

        if(silent==False or show_plot==True):
            Ya = np.zeros([num_env], dtype=np.object)
            Yb = np.zeros([num_env], dtype=np.object)
            times_new = np.linspace(min(times), max(times), 100)
        else:
            Ya = None
            Yb = None

        for i in range(len(unique_env)):
            # fit higher order model to compute derivatives (based on sm.smooth)
            Ylist[i] = DmatY[env==unique_env[i],:].reshape([-1])
            len_env = sum(env==unique_env[i])
            envtimes[i] = np.array(list(times)*len_env)
            fit = constrained_smoothspline(Ylist[i],
                                          envtimes[i],
                                          pen_degree[1],
                                          constraint="none",
                                          times_new=times_new,
                                          num_folds=num_folds,
                                          lambd="optim")
            if(silent==False or show_plot==True):
                Yb[i] = fit["smooth_vals_new"]

            dYlist[i] = np.array(list(fit["smooth_deriv"])*len_env)
            # compute differences for intergrated model fit
            if integrated_model==True:
                dYlist[i] = DmatY[env==unique_env[i],1:] - DmatY[env==unique_env[i],:-1]

            if(silent==False or show_plot==True):
                Ya[i] = fit["smooth_vals_new"]

            lambd[i] = fit["pen_par"]
            if set_initial==True:
                initial_values[i] = fit["smooth_vals"][0,0]

            else:
                initial_values[i] = None

            RSS_A[i] = sum(fit["residuals"]**2)
            UpDown_A[i] = fit["smooth_vals"][-1]
            RSS3_A[i] = None

    else:
        # initialize variables
        Ylist = np.zeros([n], dtype=np.object)
        dYlist = np.zeros([n], dtype=np.object)
        RSS_A = np.zeros([n], dtype=np.object)
        UpDown_A = np.zeros([n])
        RSS3_A = np.zeros([n])
        lambd = np.zeros([n])
        initial_values = np.zeros([n])
        times_new = np.array([0])
        if (silent==False or show_plot==True):
            Ya = np.zeros([n], dtype=np.object)
            Yb = np.zeros([n], dtype=np.object)
            times_new = np.linspace(min(times), max(times), 100)

        else:
            Ya = None
            Yb = None

        for i in range(n):
            # fit higher order model to compute derivatives (based on sm.smooth)
            Ylist[i] = DmatY[i,:]
            na_ind = np.isnan(Ylist[i])
            fit = constrained_smoothspline(Ylist[i][~na_ind],
                                          times[~na_ind],
                                          pen_degree[0],
                                          constraint="none",
                                          times_new=times_new,
                                          num_folds=num_folds,
                                          lambd="optim")
            if(silent==False or show_plot==True):
                Yb[i] = fit["smooth_vals_new"]

            dYlist[i] = fit["smooth_deriv"]
            
            # compute differences for intergrated model fit
            if integrated_model==True:
                dYlist[i] = Ylist[i][1:] - Ylist[i][:-1]
                
            # fit the reference model and compute penalty par and RSS
            if(pen_degree[0] != pen_degree[1]):
                fit = constrained_smoothspline(Ylist[i][~na_ind],
                                            times[~na_ind],
                                            pen_degree[1],
                                            constraint="none",
                                            times_new=times_new,
                                            num_folds=num_folds,
                                            lambd="optim")

            if(silent==False or show_plot==True):
                Ya[i] = fit["smooth_vals_new"]

            lambd[i] = fit["pen_par"]
            if set_initial==True:
                initial_values[i] = fit["smooth_vals"][0,0]

            else:
                initial_values[i] = None


            RSS_A[i] = sum(fit["residuals"]**2)
            UpDown_A[i] = fit["smooth_vals"][-1]
            RSS3_A[i] = sum(fit["residuals"][[0, int(len(fit["residuals"])/2)-1, len(fit["residuals"])-1]]**2)

            
    ######################################
    #
    # Step 3: Pre-compute models 
    #
    ######################################

    # convert models to modelstot
    if type(models[0])==list:
        modelstot = models

    else:
        modelstot = []
        for k in range(len(models)):
            modelstot.append(models[k])

    # intialize variables
    num_models = len(modelstot)
    data_list = np.zeros([num_models], dtype=np.object)
    varlist = np.zeros([num_models], dtype=np.object)
    data_list2 = np.zeros([num_models], dtype=np.object)

    # iteration over all potential models
    for model in range(num_models):
        if len(np.array(modelstot[model]))==0:
            model_index = []
        else:
            # for utility
            match = lambda pattern, matched: np.arange(len(matched))[np.equal(
                    np.array([[]]+list(pattern))[1:].reshape([-1,1]), 
                    np.array([[]]+list(matched))[1:].reshape([1,-1])
                ).sum(axis=0)>0]
            model_index = match(modelstot[model], ordering)
            
    ##}

    ## collect predictors X
        Xlist = np.zeros([n], dtype=np.object)
        num_pred = len(modelstot[model])+include_intercept
        if num_pred==0:
            num_pred = 1
            for i in range(n):
                Xlist[i] = np.ones([L,1])
        else:
            for i in range(n):
                Xlist[i] = Dlist[i][:, model_index]
                if include_intercept==True:
                    Xlist[i] = np.concatenate([Xlist[i], np.ones([L,1])], axis=1)
        data_list[model] = Xlist

        # compute predictors for integrated model fit
        if integrated_model==True:
            Xlist2 = np.zeros([n], dtype=np.object)
            num_pred = len(modelstot[model])+include_intercept
            if num_pred==0:
                num_pred = 1
                for i in range(n):
                    Xlist2[i] = np.ones([L-1,1])          
            else:
                for i in range(n):
                    Xlist2[i] = Dlist[i][:, model_index]
                    tmp = (Xlist2[i][0:(L-1),:] + Xlist2[i][1:L,:]) / 2
                    Xlist2[i] = tmp * np.diff(times).reshape(L-1, 1)
                    if include_intercept==True:
                        Xlist2[i] = np.concatenate([Xlist2[i], np.ones([L-1,1])], axis=1)
            data_list2[model] = Xlist2

    ######################################
    #
    # Step 4: Compute score
    #
    ######################################

    # Iterate over all models and compute score
    scores = np.zeros([num_models])
    for model in range(num_models):
        Xlist = data_list[model]
        if 'Xlist2' in locals():
            Xlist2 = data_list2[model]
        # output
        if(silent==False):
            print("Scoring model "+str(modelstot[model]))

        ## Compute constraint and RSS on constrained model
        if sample_splitting=="none":
            ###
            # Without sample splitting
            ###
            unique_env = np.array(list(set(splitting_env)))
            num_env = len(unique_env)
            RSS_B = np.zeros([len(RSS_A)])
            UpDown_B = np.zeros([len(RSS_A)])
            RSS3_B = np.zeros([len(RSS_A)])
            X = np.concatenate(list(Xlist))
            Xpred = np.concatenate(list(Xlist))
            dY = np.concatenate(list(dYlist)).reshape([-1])
            subenv_ind = [np.array(list(range(sum(splitting_env==unique_env[k]))) * L) for k in range(num_env)]
            loo_ind = np.repeat(list(splitting_env),L)
            # adjust for missing obs in integrated model fit
            if 'Xlist2' in locals(): # check if 'Xlist2' is defined
                X = np.concatenate(Xlist2)
                loo_ind2 = loo_ind
                loo_ind = np.array((splitting_env)*(L-1))
            count = 0
            ### PLOT
            if show_plot==True:
                minyplot = np.zeros([len(RSS_A)])
                maxyplot = np.zeros([len(RSS_A)])
                constrained_fit = np.zeros([len(RSS_A)], dtype=np.object)

        # Fit model on all data
            # Classical OLS regression (main effects and interactions)
            if regression_class=="OLS":
                fit = LinearRegression(fit_intercept=False).fit(X,dY)
                # remove coefficients resulting from singular fits (perfectly correlated predictors)
                coefs = fit.coef_
                coefs[np.isnan(coefs)] = 0
                fitted_dY = Xpred @ coefs

            # OLS regression with sign constraints on parameters
            elif regression_class=="signed_OLS":
                len_model = len(modelstot[model][0])
                tmp = np.array([v[0] for v in modelstot[model][0]])
                ind = np.ones([len_model])
                ind[tmp==target] = -1
                # define quadProg parameters
                bvec = np.zeros([len_model])
                Amat = diag(ind)
                dvec = dY.reshape([1, nrow(X)]) @ X
                Dmat = (X.T @ X)
                fit = solve_qp(Dmat, dvec, Amat, bvec, meq=0)
                coefs = fit.x
                fitted_dY = Xpred @ coefs

            # OLS regression with pruning based on score
            elif regression_class=="optim":
                # define loss function
                def loss_fun(beta, Y, X, env_vec, ind):
                    coefs = (10**beta)*ind
                    tmp_vec = (Y - X @ coefs)**2
                    return tmp_vec.reshape([len(env_vec),-1]).mean(axis=1).max(axis=0)

                # compute starting value using quadratic program
                len_model = len(modelstot[model][0])
                tmp = np.array([x[0] for x in modelstot[model][0]])
                ind = rep(1, len_model)
                ind[tmp==target] = -1
                bvec = np.array([10**(-10)] * len_model)
                Amat = np.diag(ind)
                dvec = dY.reshape([1, X.shape[0]]) @ X
                Dmat = (X.T @ X)
                fit = solve_qp(Dmat, dvec, Amat, bvec, meq=0)
                coefs = fit.x

                # perform optimization
                opt_res = minimize_scalar(fun = lambda beta:loss_fun(beta, dY, X, splitting.env, ind),
                                bracket = [-10,log(coefs*ind)/log(10),5], 
                                method = "Golden")
                coefs = (10**opt_res.x)*ind
                fitted_dY = Xpred @ coefs

            # Random forest regression
            elif regression_class=="random_forest":
                fit = RandomForestRegressor().fit(X,dY)
                fitted_dY = fit.predict(Xpred)

            # Wrong regression_class
            else:
                raise(Exception("Specified regression_class does not exist_ Use OLS, OLS_prune or random_forest_"))
        # fitting section ends here
        # compute score using splitting environment
            if smooth_Y==False:
                for i in range(num_env):
                    num_reps = sum(splitting_env==unique_env[i])
                    env_ind = (loo_ind2==unique_env[i])
                    for j in range(num_reps):
                        fitted_dY_tmp = fitted_dY[env_ind][subenv_ind[i]==j]
                        fit = constrained_smoothspline(Ylist[count],
                                                    times,
                                                    pen_degree[1],
                                                    constraint="fixed",
                                                    derivative_values=fitted_dY_tmp,
                                                    initial_value=initial_values[count],
                                                    times_new=times_new,
                                                    num_folds=num_folds,
                                                    lambd=lambd[count])

                        RSS_B[count] = sum(fit["residuals"]**2)
                        UpDown_B[count] = fit["smooth_vals"][len(fit["smooth_vals"])]
                        RSS3_B[count] = sum(fit["residuals"][[0, int(len(fit["residuals"])/2), len(fit["residuals"])]]**2)

                        ### PLOT
                        if show_plot==True:
                            idx = np.arange(len(splitting_env))[splitting_env==unique_env[i]][j]
                            constrained_fit[idx] = fit["smooth_vals_new"]
                        count += 1
                        
            else:
                for i in range(num_env):
                    num_reps = sum(splitting_env==unique_env[i])
                    env_ind = (loo_ind2==unique_env[i])
                    fitted_dY_tmp = fitted_dY[env_ind].reshape([L,-1]).mean(axis=1)

                    fit = constrained_smoothspline(Ylist[i],
                                              np.array(list(times)*num_reps),
                                              pen_degree[1],
                                              constraint="fixed",
                                              derivative_values=fitted_dY_tmp,
                                              initial_value=initial_values[i],
                                              times_new=times_new,
                                              num_folds=num_folds,
                                              lambd=lambd[i])

                    RSS_B[i] = sum((np.array(list(fit["smooth_vals"])*num_reps)-Ylist[i])**2)
                    UpDown_B[i] = fit["smooth_vals"][len(fit["smooth_vals"])-1]
                    RSS3_B[i] = None
                    if show_plot==True:
                        constrained_fit[i] = fit["smooth_vals_new"]

            ### PLOT 
            if show_plot==True:
                if smooth_Y==False:
                    for i in range(num_env):
                        env_ind = (splitting_env == unique_env[i])
                        Y1plot = Ylist[env_ind]
                        times1 = np.array(list(times)*sum(env_ind))
                        # for utility
                        min_ = lambda ary_obj:min([min(obj) for obj in ary_obj])
                        max_ = lambda ary_obj:max([max(obj) for obj in ary_obj])
                        miny = min(min_(tmp) for tmp in [[Y1plot], Ya[env_ind], Yb[env_ind], constrained_fit[env_ind]])
                        maxy = max(max_(tmp) for tmp in [[Y1plot], Ya[env_ind], Yb[env_ind], constrained_fit[env_ind]])
                        
                        # plot
                        plt.plot(times1, Y1plot,'o',c='black')
                        plt.xlabel("times")
                        plt.ylabel("concentration")
                        plt.ylim([miny, maxy])
                        which_ind = np.arange(len(splitting_env))[env_ind]
                        for k in which_ind:
                            plt.plot(times_new, Ya[k], 'l', c="red")
                            plt.plot(times_new, Yb[k], 'l', c="blue")
                            plt.plot(times_new, constrained_fit[[k]], 'l', c="green")
                        #readline("Press enter")
                        plt.show()

                else:
                    for i in range(num_env):
                        env_ind = (splitting_env == unique_env[i])
                        times1 = np.array(list(times)*sum(env_ind))
                        L = len(times)
                        min_ = lambda ary_obj:min([min(obj) for obj in ary_obj])
                        max_ = lambda ary_obj:max([max(obj) for obj in ary_obj])
                        miny = min(min_(tmp) for tmp in [Ylist, [Ya[i]], [Yb[i]], [constrained_fit[i]]])
                        maxy = max(max_(tmp) for tmp in [Ylist, [Ya[i]], [Yb[i]], [constrained_fit[i]]])
                        
                        # plot
                        plt.plot(times1, Ylist[i],'o',c='black')
                        plt.xlabel("times")
                        plt.ylabel("concentration")
                        plt.ylim([miny, maxy])

                        for k in range(sum(env_ind)):
                            plt.plot(times_new, Ya[i], 'l', c="red")
                            plt.plot(times_new, Yb[i], 'l', c="blue")
                            plt.plot(times_new, constrained_fit[[i]], 'l', c="green")
                        #readline("Press enter")
                        plt.show()
                    
        elif sample_splitting=="loo":
            ###
            # With leave-one-out sample splitting
            ###
            unique_env = np.array(list(set(splitting_env)))
            num_env = len(unique_env)
            RSS_B = np.zeros([len(RSS_A)])
            UpDown_B = np.zeros([len(RSS_A)])
            RSS3_B = np.zeros([len(RSS_A)])
            X = np.concatenate(Xlist)
            dY = np.array(list(dYlist)).reshape([-1])
            
# fixed from the following line, might be the cause of bug
#           subenv_ind = [np.array(list(range(sum(splitting_env==unique_env[k])))*L) for k in range(num_env)]
            subenv_ind = [np.repeat(list(range(sum(splitting_env==unique_env[k]))),L) for k in range(num_env)]
            loo_ind = np.repeat(list(splitting_env), L)
            # adjust for missing obs in integrated model fit
            if 'Xlist2' in locals(): # check if 'Xlist2' is defined
                X2 = X                            
                X = np.concatenate(Xlist2)
                loo_ind2 = loo_ind
                loo_ind = np.array(list(splitting_env)*(L-1))  
            count = 0

            ### PLOT
            if show_plot==True:
                minyplot = np.zeros([len(RSS_A)])
                maxyplot = np.zeros([len(RSS_A)])
                constrained_fit = np.zeros([len(RSS_A)], dtype=np.object)

            for i in range(num_env):
                # compute derivative constraint using OLS
                dYout = dY[loo_ind!=unique_env[i]]
                Xout = X[loo_ind!=unique_env[i],:]
                env_out = loo_ind[loo_ind!=unique_env[i]]
                Xin = X[loo_ind==unique_env[i],:]
                # adjust for missing vlaue in integrated model fit
                if 'Xlist2' in locals(): # check if 'Xlist2' is defined
                    Xin = X2[loo_ind2==unique_env[i],:]
                # dealing with na
                Xout_nona = Xout[np.isnan(Xout).sum(axis=1)==0,:]
                dYout_nona = dYout[np.isnan(Xout).sum(axis=1)==0]
                env_out = env_out[np.isnan(Xout).sum(axis=1)==0]

                # Classical OLS regression (main effects and interactions)
                if regression_class=="OLS":
                    fit = LinearRegression(fit_intercept=False).fit(Xout_nona, dYout_nona)
                    # remove coefficients resulting from singular fits (perfectly correlated predictors)
                    coefs = fit.coef_
                    coefs[np.isnan(coefs)] = 0
                    fitted_dY = Xin @ coefs

                # OLS regression with sign constraints on parameters
                elif regression_class=="signed_OLS":
                    len_model = len(modelstot[model][0])
                    tmp = np.array([x[0] for x in modelstot[model][0]])
                    ind = np.ones([len_model])
                    ind[tmp==target] = -1
                    # define quadProg parameters
                    bvec = np.zeros([len_model])
                    Amat = diag(ind)
                    dvec = dYout_nona.reshape([1, Xout_nona.shape[0]]) @ Xout_nona
                    Dmat = Xout_nona.T @ Xout_nona
                    fit = solve_qp(Dmat, dvec, Amat, bvec, meq=0)
                    coefs = fit.x
                    fitted_dY = Xin @ matrix(coefs, ncol(Xin), 1)

                # OLS regression with pruning based on score
                elif(regression_class=="optim"):
                    # define loss function
                    def loss_fun(beta, Y, X, env_vec, ind):
                        coefs = (10**beta)*ind
                        tmp_vec = (Y - X @ coefs)**2
                        return (np.mean(tmp_vec))
                    # compute starting value using quadratic program
                    len_model = len(modelstot[model][0])
                    tmp = np.array(x[0] for x in [modelstot[model][0]])
                    ind = np.ones([len_model])
                    ind[tmp==target] = -1
                    bvec = 10**(-10) * np.ones([len_model])
                    Amat = diag(ind)
                    dvec = dYout_nona.reshape([1, nrow(Xout_nona)]) @ Xout_nona
                    Dmat = Xout_nona.T @ Xout_nona
                    fit = solve_qp(Dmat, dvec, Amat, bvec, meq=0)
                    coefs = fit.x
                    # perform optimization
                    opt_res = minimize_scalar(fun = lambda beta:loss_fun(beta, dYout_nona, Xout_nona, env_out, ind),
                                bracket = [-10,log(coefs*ind)/log(10),5], 
                                method = "Golden")

                    fitted_dY = Xin @ coefs

                # Random forest regression
                elif regression_class=="random_forest":
                    fit = RandomForestRegressor().fit(Xout_nona,dYout_nona)
                    fitted_dY = fit.predict(Xin)

                # Wrong regression_class
                else:
                    stop("Specified regression_class does not exist_ Use OLS, OLS_prune or random_forest_")

                # Fit individual models with derivative constraint
                if smooth_Y==False:
                    for j in range(sum(splitting_env==unique_env[i])):
                        na_ind = np.logical_or(
                            np.isnan(Ylist[count]),
                            np.isnan(Xin[subenv_ind[i]==j,:]>0).sum(axis=1)
                        )
    
                        fitted_dY_tmp = fitted_dY[subenv_ind[i]==j]
                        fit = constrained_smoothspline(Ylist[count][~na_ind],
                                                        times[~na_ind],
                                                        pen_degree[1],
                                                        constraint="fixed",
                                                        derivative_values=fitted_dY_tmp[~na_ind],
                                                        initial_value=initial_values[count],
                                                        times_new=times_new,
                                                        num_folds=num_folds,
                                                        lambd=lambd[count])

                        RSS_B[count] = sum(fit["residuals"]**2)
                        UpDown_B[count] = fit["smooth_vals"][len(fit["smooth_vals"])-1]
                        RSS3_B[count] = sum(fit["residuals"][[1, int(len(fit['residuals'])/2), len(fit['residuals'])-1]]**2)
                        ### PLOT
                        if show_plot==True:
                            idx = np.arange(len(splitting_env))[splitting_env==unique_env[i]][j]
                            constrained_fit[idx] = fit["smooth_vals_new"]
                        count +=1
                        
                else:
                    len_env = sum(splitting_env==unique_env[i])
                    fitted_dY_tmp = fitted_dY.reshape([L,-1]).mean(axis=1)                    
                    fit = constrained_smoothspline(Ylist[i],
                                                  times,
                                                  pen_degree[1],
                                                  constraint="fixed",
                                                  derivative_values=fitted_dY_tmp,
                                                  initial_value=initial_values[i],
                                                  times_new=times_new,
                                                  num_folds=num_folds,
                                                  lambd=lambd[i])
                    RSS_B[i] = sum(fit['residuals']**2)
                    UpDown_B[i] = fit["smooth_vals"][len(fit["smooth_vals"])-1]
                    RSS3_B[i] = None
                    if show_plot==True:
                        constrained_fit[i] = fit["smooth_vals_new"]

            ### PLOT
            if show_plot==True:
                if smooth_Y==False:
                    for i in range(num_env):
                        env_ind = (splitting_env == unique_env[i])
                        Y1plot = np.array(list(Ylist[env_ind])).reshape([-1])
                        times1 = np.array(list(times)*sum(env_ind))
                        # for utility
                        min_ = lambda ary_obj:min([min(obj) for obj in ary_obj])
                        max_ = lambda ary_obj:max([max(obj) for obj in ary_obj])
                        miny = min(min_(tmp) for tmp in [[Y1plot], Ya[env_ind], Yb[env_ind], constrained_fit[env_ind]])
                        maxy = max(max_(tmp) for tmp in [[Y1plot], Ya[env_ind], Yb[env_ind], constrained_fit[env_ind]])

                        # plot
                        plt.plot(times1, Y1plot,'o',c='black')
                        plt.xlabel("times")
                        plt.ylabel("concentration")
                        plt.ylim([miny, maxy])

                        which_ind = np.arange(len(splitting_env))[env_ind]
                        for k in which_ind:
                            plt.plot(times_new, Ya[k], '-', c="red")
                            plt.plot(times_new, Yb[k], '-', c="blue")
                            plt.plot(times_new, constrained_fit[k], '-', c="green")
                        #print(str(Ya))
                        #readline("Press enter")
                        plt.show()

                else:
                    env_ind = splitting_env == unique_env[i]
                    times1 = np.array(list(times)*sum(env_ind))
                    L = len(times)
                    # for utility
                    min_ = lambda ary_obj:min([min(obj) for obj in ary_obj])
                    max_ = lambda ary_obj:max([max(obj) for obj in ary_obj])
                    miny = min(min_(tmp) for tmp in [Ylist, [Ya[i]], [Yb[i]], [constrained_fit[i]]])
                    maxy = max(max_(tmp) for tmp in [Ylist, [Ya[i]], [Yb[i]], [constrained_fit[i]]])
                    # plot
                    plt.plot(times1, Ylist[i],'o',c='black')
                    plt.xlabel("times")
                    plt.ylabel("concentration")
                    plt.ylim([miny, maxy])

                    for k in range(sum(env_ind)):
                        plt.plot(times_new, Ya[k], '-', c="red")
                        plt.plot(times_new, Yb[k], '-', c="blue")
                        plt.plot(times_new, constrained_fit[k], '-', c="green")
                    #readline("Press enter")
                    plt.show()
                    
                    
                    for i in range(num.env):
                        env_ind = splitting_env == unique_env[i]
                        times1 = np.array(list(times)*sum(env_ind))
                        L = len(times)
                        # for utility
                        min_ = lambda ary_obj:min([min(obj) for obj in ary_obj])
                        max_ = lambda ary_obj:max([max(obj) for obj in ary_obj])
                        miny = min(min_(tmp) for tmp in [Ylist, [Ya[i]], [Yb[i]], [constrained_fit[i]]])
                        maxy = max(max_(tmp) for tmp in [Ylist, [Ya[i]], [Yb[i]], [constrained_fit[i]]])
                        # plot
                        plt.plot(times_new, Ya[k], '-', c="red")
                        plt.plot(times_new, Yb[k], '-', c="blue")
                        plt.plot(times_new, constrained_fit[k], '-', c="green")
                    #readline("Press enter")
                    plt.show()

        else:
            raise(Exception("Specified sample_splitting does not exist_ Use none or loo_"))

        ## compute score
        if score_type=="max": 
            score = np.max((RSS_B-RSS_A)/RSS_A)

        elif score_type=="mean":
            score = np.mean((RSS_B-RSS_A)/RSS_A)
# for debug
            #print(RSS_A)
            #print(RSS_B)
            #print(RSS_B-RSS_A)
        elif score_type=="mean2":
            score = np.mean(RSS_B)

        elif score_type=="mean_weighted":
            score = np.mean(weight_vec*(RSS_B-RSS_A)/RSS_A)

        elif score_type=="mean_3point":
            score = np.mean((RSS3_B-RSS3_A)/RSS3_A)

        elif score_type=="second-worst":
            tmp = (RSS_B-RSS_A)/RSS_A
            tmp.sort(order='decreasing')
            score = tmp[1]

        elif score_type=="max-mean":
            if smooth_Y==True:
                score = np.max((RSS_B-RSS_A)/RSS_A)

            else:
                uenv = list(set(env))
                n_env = len(uenv)
                tmp = np.zeros([n_env])
                for i in range(n_env):
                    tmp[i] = np.mean((RSS_B[env==uenv[i]]-RSS_A[env==uenv[i]])/RSS_A[env==uenv[i]])

                score = np.max(tmp)

        elif score_type=="updown1":
            score = np.max(np.abs(UpDown_A-UpDown_B))

        elif score_type=="updown2":
            score = np.mean(np.abs(UpDown_A-UpDown_B))

        else:
            raise(Exception("Specified score_type does not exist_ Use max, mean or max-mean_"))

        # Output
        if silent==False:
            print("Model has a score of {}".format(score))

        scores[model] = score


    ## Return results
    return(scores)
