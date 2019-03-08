import numbers
import numpy as np


def generate_data_targetmodel(env=np.ones([10]),
                          noise_var=0.01,
                          L=15,
                          d=7,
                          seed=None):

    assert(type(env)==np.ndarray)
    
    # set seed
    if type(seed)==numbers.Number:
        np.random.seed(seed)

    # read out parameters
    n = 10000
    n_env = len(set(env))
    reps = len(env)
    time = np.linspace(0, 10, L)
    time_index = np.linspace(0, n-1, L).astype(int)
    simulated_data = np.zeros([reps, d*L])
    target = d-1
    _,env_size = np.unique(env, return_counts=True)

    # Define function to generate random smooth functions
    def smooth_fun(par, n):
        fun = lambda t: par[0]/(1+np.exp(par[1]*t))+par[2]/(1+np.exp(par[3]*t))
        tvec = np.linspace(-3,3, n)
        fvec = fun(tvec)
        return(fvec)

    # Generate data
    for a in range(n_env):
        current_env = np.unique(env)[a]
        
        # Generate random smooth predictor functions
        Xmat = np.zeros([n, d-1])
        for i in range(d-1):
            Xmat[:, i] = smooth_fun(np.random.randn(4), n)

        # read out data for predictors
        noise_var2 = np.apply_along_axis(arr=Xmat, axis=0, func1d=lambda x:noise_var*(max(x)-min(x)))
        noiseterm = np.random.randn(L*(d-1)*env_size[a]).reshape([env_size[a], (d-1), L]) * noise_var2.reshape([1,d-1,1])
        noiseterm = noiseterm.reshape([env_size[a], L*(d-1)])
        simulated_data[env==current_env,:(d-1)*L] \
            = np.array([Xmat[time_index,:].T.flatten() for i in range(env_size[a])]) + noiseterm
        # use numerical integration to generate Y
        Y1 = np.insert(np.cumsum(0.5*(Xmat[0:(n-1), 0] + Xmat[1:n, 0])), 0, 0)
        Y2 = np.insert(np.cumsum(0.5*(Xmat[0:(n-1), 1] + Xmat[1:n, 1])), 0, 0)
        Y = (0.1*Y1+0.2*Y2)*0.001
        noise_var2 = noise_var*(max(Y)-min(Y))
        noiseterm = np.random.randn(L*env_size[a]).reshape([env_size[a], L]) * noise_var2
        simulated_data[env==current_env, (d-1)*L:] \
            = np.array([Y[time_index] for i in range(env_size[a])]) + noiseterm
        #plt.plot(np.arange(len(Y[time_index])), Y[time_index])
        
    return {"simulated_data":simulated_data,
          "time":time,
          "env":env,
          "target":target,
          "true_model":[[0], [1], [0, 1]]}