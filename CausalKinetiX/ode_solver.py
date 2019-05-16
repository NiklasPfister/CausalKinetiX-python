"""
Solves a Mass-action ODE for a target variable Y by using smooth approximations of the predictor variables X.

Parameters
----------
time_vec : 1d-array. float. Specifices the points at which to evaluate the target trajectory.
initial_value : float. Specifies the value of the target at initial time point.
times :1d-array of length L. flaot. Specifices the time points at which the predictors where observed.
X : predictor matrix(2d-array) of dimension L x d. Each column corresponds to a different predictor observed at the time points in "times"
model : list of mass-action terms. Each element in the list consists of a vector of predictor variables which are multiplied to a single term in the mass-action equation. for example, [[0],[1,2]] stands for a model whose predictor variables are x0 and x1*x2.
target : integer specifing which variable is the target.
coefs : 1d-array. float. Specifies the parameter values for each term in "model".
included_vars : 1d-array of variables 1 to d. Can be used in order to save computational costs if not all variables given in "X" actually show up in "model".
smooth_type : string. Specifies which type of smoothing to use. Following options exist: "smoothing.spline", "linear", "constant". "loess" is not currently supported in Python version.
reltol : float. Relative tolarance used in CVODE.
abstol : float. Absolute tolarance used in CVODE.


Returns
-------
matrix (2d-array) with 2 columns. the first column contain time points at which target values are evaluated. second column contains the target values at each time points given by the first column.

Examples
--------
## Generate data from Maillard reaction
## Generate data from Maillard reaction
simulation_obj = generate_data_maillard(target=1, noise=0.5, noise_target=10,
                                         env=np.array(list(range(5))*3),
                                         L=20)
D = simulation_obj["simulated_data"]
time = simulation_obj["time"]
env = simulation_obj["env"]
target = simulation_obj["target"]
true_model = simulation_obj["true_model"]

# coefficients of the maillard model with "target=0"
theta=np.array([0.01, 0.00509, 0.00047, 0.0011, 0.00712, 0.00439, 0.00018, 0.11134, 0.14359, 0.00015, 0.12514])
coef=[theta[0], -(theta[1]+theta[3]+theta[4]), -theta[9]]

# compute the target trajectory from ode model given other observations
odefit = ode_solver(time_vec=time,
                    initial_value=X_[0,[target]],
                    times=time,
                    X=X[:len(time),:],
                    model=true_model,
                    target=target,
                    coefs=coef)

# plot the trajectory and observatons
plt.plot(odefit[:,0], odefit[:,1], '-')
plt.plot(time, X[:len(time),target], 'o')

Notes
-----
For further details see the following references.
Pfister, N., S. Bauer, J. Peters (2018).
Identifying Causal Structure in Large-Scale Kinetic Systems
(https://arxiv.org/pdf/1810.11776.pdf)
"""

import numpy as np
import scipy.integrate
# from scipy.interpolate import interp1d
# import statsmodels.nonparametric.smoothers_lowess
from .constrained_smoothspline import constrained_smoothspline


def ode_solver(time_vec, initial_value, times, X, model,
               target, coefs, included_vars=None,
               smooth_type="smoothing_spline",
               reltol=1e-10, abstol=1e-16):

    if included_vars is None:
        included_vars = np.arange(X.shape[1])

    ## Fit spline on each predictor
    if smooth_type == "smoothing_spline":
        splinefun = [
            constrained_smoothspline(y=X[:, j],
                                     times=times,
                                     pen_degree=2,
                                     times_new=times,
                                     constraint="none")['trained_fit']
            for j in included_vars
        ]
        splinefun = [lambda t, fit=fit: fit.predict(np.array([t])) for fit in splinefun]
        # without fit=fit, error occures,
        # for detail, go to https://docs.python-guide.org/writing/gotchas/#late-binding-closures

    # Local Polynomial Regression is not currently supported
    # no popular package of python supports Local Polynomial Regression
    #
    # elif smooth_type == "loess":
    #     splinefun = lapply(1:len(included_vars), function(j) loess( ~ times, span=0.50))
    #     splinefun = lapply(splinefun, function(fit){function(t) predict(fit, t)})

    elif smooth_type == "linear":
        splinefun = [
            scipy.interpolate.interp1d(times, X[:, j], kind="linear")
            for j in included_vars
        ]

    elif smooth_type == "constant":
        splinefun = [
            scipy.interpolate.interp1d(times, X[:, j], kind="nearest")
            for j in included_vars
        ]

    ## Construct RHS
    def odefun(t, y):
        deriv = 0
        for term in range(len(model)):
            tmp = 1
            for var in model[term]:
                if var == target:
                    tmp = tmp*y
                else:
                    tmp = tmp*splinefun[int(
                        np.arange(len(included_vars))[var == included_vars])](t)
                ####
            ####
            deriv += coefs[term]*tmp
        return(deriv)

    ## Solve ODE
    result = scipy.integrate.solve_ivp(y0=initial_value,
                                       fun=odefun,
                                       t_span=[time_vec.min(), time_vec.max()],
                                       t_eval=time_vec,
                                       rtol=reltol, atol=abstol)
    odefit = np.concatenate([result.t.reshape([-1, 1]), result.y.T], axis=1)

    return odefit
