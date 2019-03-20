"""
Generate sample data from the hidden variable model.

Parameters
----------

env : integer 1d-array of length n encoding to which experiment each repetition belongs. be aware that variable index starts from 0.
L : number of time points for evaluation.
intervention : type of intervention.
intervention_par : parameters used in the interventions.
hidden : boolean whether the variables H1 and H2 should be removed from output.
ode.solver : specifies which ODE solver to use when solving ODE.
seed : random seed.
silent : set to FALSE if status output should be produced.

Returns
-------
dict object with following keys and values
"simulated_data" : D-matrix(2d-array) of noisy data.
"time" : 1d-array containing time points.
"env" : specifying the experimental environment.
"simulated_model" : object returned by ODE solver.
"true_model" : 1d-array specifying the target equation model.
"target" : target variable.

Examples
--------
## Generate data from Maillard reaction
>>> simulation_obj = generate_data_hidden(env=np.array(list(range(5))*3),
                                          L=15,
                                          noise=0.02,
                                          only_target_noise=False,
                                          relativ=True,
                                          intervention="initial_blockreactions5",
                                          intervention_par=0.2)
>>> D = simulation_obj["simulated_data"]
>>> time = simulation_obj["time"]
>>> env = simulation_obj["env"]
>>> target = simulation_obj["target"]
>>> # plot the observation of target variable in environment 0
>>> plt.plot(time, D[0,-len(time):], '-',c="black")
>>> plt.plot(time, D[0,-len(time):], 'o',c="red")
>>> plt.legend(["observations"])

Notes
-----
The function CausalKinetiX_modelranking can be used if the variable ranking is not required.
For further details see the following references.
Pfister, N., S. Bauer, J. Peters (2018).
Identifying Causal Structure in Large-Scale Kinetic Systems
(https://arxiv.org/pdf/1810.11776.pdf)
"""

import numpy as np
import scipy.integrate
import numbers

def generate_data_hidden(env=np.zeros(10),
                         L=15,
                         noise=0.01,
                         only_target_noise=True,
                         relativ=False,
                         intervention="initial_blockreactions5",
                         hidden=True,
                         intervention_par=None,
                         ode_solver="LSODA",
                         seed=None,
                         silent=False):
    # set seed
    if(type(seed)==numbers.Number):
        np.random.seed(seed)
    if intervention_par==None:
        print("please_spcify intervention_par")

    ######################################
    #
    # Simulate data
    #
    ######################################


    ###
    # Initialize RHS
    ###

    def reactions(t, x, theta):

        dx1 = theta[2]*x[6] - theta[0]*x[0] - theta[5]*x[0]*x[3]
        dx2 = theta[1]*x[6] - theta[6]*x[1]
        dx3 = theta[5]*x[0]*x[3] - theta[4]*x[2] + theta[7]*x[4]
        dx4 = theta[4]*x[2] - theta[5]*x[0]*x[3] - theta[8]*x[3]
        dx5 = theta[8]*x[3] - theta[7]*x[4]
        dx6 = theta[6]*x[1]
        dx7 = theta[0]*x[0] - (theta[1]+theta[2])*x[6]
        dx8 = theta[1]*x[6] - theta[3]*x[7]
        dx9 = theta[3]*x[7] + theta[4]*x[2]

        return(np.array([dx1,dx2,dx3,dx4,dx5,dx6,dx7,dx8,dx9]))

    ###
    # Set parameters
    ###

    d = 9
    target = 8
    time_grid = np.linspace(0, 100, int(100/0.005)+1)
    """
    Commented out
    -------------
    # in R implementation, log-scale is used
    time_index = np.zeros([L])
    time_index[0] = 1
    time_index[1:] = np.exp(np.linspace(0, np.log(len(time_grid)), (L-1)))-1
    time_index = time_index.astype(np.int)
    """
    time_index = (np.linspace(1, len(time_grid)**(1/2), L)**2).astype(int)-1
    initial_obs = np.array([5, 0, 0, 5, 0, 0, 0, 0, 0])
    theta_obs = np.array([0.8, 0.8, 0.1, 1, 0.03, 0.6, 1, 0.2, 0.5])*0.1
    n_env = len(set(env))

    fixed_reactions = np.array([4, 5, 7])

    true_set = [[2]]

    # for utility
    match = lambda pattern, matched: np.equal(
            np.array(pattern).reshape([-1,1]), 
            np.array(matched).reshape([1,-1])
        ).sum(axis=0)>0
    
    
    ###
    # Define interventions
    ###

    if intervention == "only_initial":
        def intervention_fun():
            initial_int = np.zeros(7)
            initial_int[0] = np.random.rand(1)*10
            initial_int[3] = np.random.rand(1)*10
            return({"initial":initial_int,
                    "theta":theta_obs})

    elif intervention == "only_blockreactions":
        def intervention_fun():
            r_vec = ~match(fixed_reactions, np.arange(len(theta_obs)))
            num_reactions = int(sum(r_vec))
            theta_int = theta_obs
            theta_int[r_vec] = theta_int[r_vec]*np.random.binomial(1,1-1/num_reactions, size=num_reactions)
            return({"initial":initial_obs,
                    "theta":theta_int})

    elif intervention == "initial_blockreactions1":
        true_set = [[1],[2]]
        def intervention_fun():
            r_vec = ~match(fixed_reactions, np.arange(len(theta_obs)))
            num_reactions = int(sum(r_vec))
            theta_int = theta_obs
            theta_int[r_vec] = theta_int[r_vec]*np.random.binomial(1,1-2/num_reactions, size=num_reactions)
            initial_int = np.zeros(8)
            initial_int[0] = np.random.rand(1)*10
            initial_int[3] = np.random.rand(1)*10
            initial_int[4] = np.random.rand(1)*10
            return({"initial":initial_int,
                    "theta":theta_int})

    elif intervention == "initial_blockreactions2":
        true_set = [[2]]
        theta_obs[6] = np.random.rand(1)
        fixed_reactions = [3, 4]
        def intervention_fun():
            r_vec = ~match(fixed_reactions, np.arange(len(theta_obs)))
            num_reactions = int(sum(r_vec))
            theta_int = theta_obs
            theta_int[r_vec] = theta_int[r_vec]*np.random.binomial(1,1-2/num_reactions, size=num_reactions)
            initial_int = np.zeros(8)
            initial_int[0] = np.random.rand(1)*10
            initial_int[3] = np.random.rand(1)*10
            initial_int[4] = np.random.rand(1)*10
            return({"initial":initial_int,
                    "theta":theta_int})

    elif intervention == "initial_blockreactions3":
        true_set = [[2]]
        const = intervention_par
        theta_obs[6] = theta_obs[3]+(np.random.rand(1)-1/2)*2*const
        fixed_reactions = np.array([3, 4])
        def intervention_fun():
            r_vec = ~match(fixed_reactions, np.arange(len(theta_obs)))
            num_reactions = int(sum(r_vec))
            theta_int = theta_obs
            theta_int[r_vec] = theta_int[r_vec]*np.random.binomial(1,1-2/num_reactions, size=num_reactions)
            if r_vec[6]==True:
                theta_int[6] = theta_obs[3]+(np.random.rand(1)-1/2)*2*const
            initial_int = np.zeros(8)
            initial_int[0] = np.random.rand(1)*10
            initial_int[3] = np.random.rand(1)*10
            initial_int[4] = np.random.rand(1)*10
            return({"initial":initial_int,
                    "theta":theta_int})

    elif intervention == "initial_blockreactions4":
        true_set = [[2]]
        theta_obs[6] = intervention_par
        fixed_reactions = [3, 4, 6]
        def intervention_fun():
            r_vec = ~match(fixed_reactions, np.arange(len(theta_obs)))
            num_reactions = int(sum(r_vec))
            theta_int = theta_obs
            theta_int[r_vec] = theta_int[r_vec]*np.random.binomial(1,1-2/num_reactions, size=num_reactions)
            initial_int = np.zeros(9)
            initial_int[0] = np.random.rand(1)*10
            initial_int[3] = np.random.rand(1)*10
            initial_int[4] = np.random.rand(1)*10
            return({"initial":initial_int,
                    "theta":theta_int})

    elif intervention == "initial_blockreactions5":
        true_set = [[2]]
        fixed_reactions = [4, 5, 7]
        def intervention_fun():
            r_vec = ~match(fixed_reactions, np.arange(len(theta_obs)))
            num_reactions = int(sum(r_vec))
            theta_int = theta_obs
            theta_int[r_vec] = theta_int[r_vec]*np.random.binomial(1,1-1/num_reactions, size=num_reactions)
            theta_int[6] = max(theta_obs[7] + (np.random.rand(1)-1/2)*2*intervention_par,0)
            #theta_int[10] = runif(1, 0, intervention_par)
            initial_int = np.zeros(9)
            initial_int[0] = np.random.rand(1)*10
            initial_int[3] = np.random.rand(1)*10
            initial_int[4] = np.random.rand(1)*10
            #initial_int[2] = runif(1, 0, intervention_par)
            return({"initial":initial_int,
                    "theta":theta_int})

    ###
    # Generate data from exact ODE and generate observations
    ###

    # set up required variables
    simulated_data = np.zeros([len(env), L*d])
    simulated_model = np.zeros([n_env], dtype=np.object)
    time = time_grid[time_index]

    # iterate over environments
    for i in range(n_env):
        env_size = sum(env == i)
        # perform intervention: initial condition
        if i == 0:
            initial = initial_obs
            theta = theta_obs
        else:
            int_data = intervention_fun()
            initial = int_data["initial"]
            theta = int_data["theta"]

        # solve ODE numerically
        if not silent==True:
            print("Currently solving ODE-system on environment {}".format(i))

        # catch warnings from ODE solver --> not a good model
        simulated_model[i] = scipy.integrate.solve_ivp(y0 = initial, 
                                                       fun = lambda t,x:reactions(t,x,theta), 
                                                       t_span=[0,100], 
                                                       t_eval=time_grid, 
                                                       method=ode_solver
                                                      ).y.T
        if type(simulated_model[i])==float or simulated_model[i].shape[0]<len(time_grid):
            if not silent==True:
                print("Problem in ode-solver")
                return(None)

        # select time points and add noise
        if only_target_noise==True:
            tmp = simulated_model[i][time_index, :]
            if relativ==True:
                noise_var = noise*(tmp[:, target].max() - tmp[:, target].min()) + 0.0000001
                noiseterm = (np.random.randn(L*env_size)*noise_var).reshape([env_size, L])
            else:
                noiseterm = (np.random.randn(L*env_size)*noise).reshape([env_size, L])
            simulated_data[env==i, :] =  np.array([list(tmp.T.flatten())]*env_size)
            simulated_data[env==i, target*L:(target+1)*L]\
                = simulated_data[env==i, target*L:(target+1)*L] + noiseterm

        else:
            tmp = simulated_model[i][time_index, :]
            if relativ==True:
                noise_var = noise*(tmp.max(axis=0) - tmp.min(axis=0)) + 0.0000001 # diff of vector,matrix?
                noiseterm = np.random.randn(L*d*env_size) 
                noiseterm = noiseterm.reshape([env_size, L*d]) * np.repeat(noise_var, L).reshape([1,-1])
            else:
                noiseterm = np.random.randn(L*d*env_size) * np.array(list(noise_var)*L*env_size)
                noiseterm = noiseterm.reshape([env_size, L*d])

            simulated_data[env==i,:] =  np.array([list(tmp.T.flatten())]*env_size) + noiseterm

    ###
    # Check if a blow-up occured
    ###

    blowup = np.zeros([n_env], dtype=np.bool)
    for i in range(n_env):
        blowup[i] = np.abs(simulated_model[i][len(time_grid)-1,-1]) > 1e+8 or\
                    np.isnan(np.abs(simulated_model[i][len(time_grid)-1,-1]))

    if sum(blowup)>0:
        if not silent==True:
            print("Detected blow-up")
        ## Call function again (no seed!)
        return(blowup)

    ###
    # Remove hidden variables
    ###

    if hidden==True:
        hidden_index = np.zeros(9*L, dtype=np.bool)
        hidden_index[5*L:7*L] = True
        simulated_data = simulated_data[:,~hidden_index]
        target = 6
    else:
        target = 8
        
    return({
        "simulated_data":simulated_data,
        "time":time,
        "env":env,
        "simulated_model":simulated_model,
        "true_model":true_set,
        "target":target,
    })