import numpy as np
import numbers
import scipy.integrate

def generate_data_maillard(target,
                           env=np.ones([10]),
                           L=15,
                           noise=0.01,
                           only_target_noise=True,
                           noise_target=1,
                           relativ=False,
                           intervention="initial_blockreactions",
                           ode_solver="LSODA",
                           seed=None,
                           silent=False
                          ):
    # set seed
    if type(seed)==numbers.Number:
        np.random.seed(seed)

    ###
    # Setting default parameters
    ###

    ######################################
    #
    # Simulate data
    #
    ######################################


    ###
    # Initialize RHS
    ###

    def reactions(t, x, theta):

        dx1 = -(theta[0]+theta[2])*x[0] + theta[1]*x[1] - theta[6]*x[0]*x[9]
        dx2 = -(theta[1]+theta[3]+theta[4])*x[1] + theta[0]*x[0] - theta[9]*x[1]*x[9]
        dx3 = theta[2]*x[0] + theta[3]*x[1]
        dx4 = 2*theta[4]*x[1] - theta[5]*x[3]
        dx5 = theta[5]*x[3] + theta[7]*x[6]
        dx6 = theta[5]*x[3]
        dx7 = -(theta[7]+theta[8])*x[6] + theta[6]*x[0]*x[9]
        dx8 = theta[8]*x[6] - theta[10]*x[7] + theta[9]*x[1]*x[9]
        dx9 = theta[2]*x[0] + theta[3]*x[1]
        dx10 = theta[7]*x[6] - theta[6]*x[0]*x[9] - theta[9]*x[1]*x[9]
        dx11 = theta[10]*x[7]    
        return(np.array([dx1,dx2,dx3,dx4,dx5,dx6,dx7,dx8,dx9,dx10,dx11]))

  
    ###
    # Set parameters
    ###
  
    d = 11
    time_grid = np.linspace(0, 100, num= int(100/0.005)+1)
    time_index = (np.linspace(1, len(time_grid)**(1/2), L)**2).astype(int)
    initial_obs = np.array([160, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0])
    theta_obs = np.array([0.01, 0.00509, 0.00047, 0.0011, 0.00712, 0.00439,
                 0.00018, 0.11134, 0.14359, 0.00015, 0.12514])
    n_env = len(set(env))

    included_reactions = [
        np.array([0, 1, 2, 6]),
        np.array([0, 1, 3, 4 ,9]),
        np.array([2, 3]),
        np.array([4, 5]),
        np.array([5, 7]),
        np.array([5]),
        np.array([6, 7, 8]),
        np.array([8, 9, 10]),
        np.array([2, 3]),
        np.array([6, 7, 9]),
        np.array([10])
    ]
    included_reactions = included_reactions[target]

    true_set = [
        [[0], [1], [0, 9]],
        [[0], [1], [1, 9]],
        [[0], [1]],
        [[1], [3]],
        [[3], [6]],
        [[3]],
        [[6], [0, 9]],
        [[6], [7], [1, 9]],
        [[0], [1]],
        [[6], [0, 9], [1, 9]],
        [[7]]
    ]
    true_set = true_set[target]  

    ###
    # Define interventions
    ###  

    if intervention == "only_initial":
        def intervention_fun():
            initial_int = np.zeros([2 + 7 + 1 + 1])
            initial_int[:2] = np.random.rand(2)*360
            initial_int[9] = np.random.rand(1)*30
            return({"initial":initial_int,
                    "theta"  :theta_obs})

    elif intervention == "only_blockreactions":
        def intervention_fun():
            num_reactions = d-len(included_reactions)
            # exclude the included_reactions from 0,...,d
            # (1:d)[-included_reactions] ... R language version
            r_vec = ~np.any(np.arange(d).reshape([-1,1])==included_reactions.reshape([1,-1]), axis=1)
            theta_int = theta_obs
            theta_int[r_vec] = theta_int[r_vec]*np.random.binomial(size=num_reactions, n=1, p=1-2/num_reactions)
            return({"initial":initial_obs,
                    "theta":theta_int})

    elif intervention == "initial_blockreactions":
        def intervention_fun():
            num_reactions = d-len(included_reactions)
            # exclude the included_reactions from 0,...,d
            # (1:d)[-included_reactions] ... R language version
            r_vec = ~np.any(np.arange(d).reshape([-1,1])==included_reactions.reshape([1,-1]), axis=1)
            theta_int = theta_obs
            theta_int[r_vec] = theta_int[r_vec]*np.random.binomial(size=num_reactions, n=1, p=1-3/num_reactions)
            initial_int = np.zeros([2 + 7 + 1 + 1])
            initial_int[:2] = np.random.rand(2)*800
            initial_int[9] = np.random.rand(1)*75

            return({"initial":initial_int,
                    "theta":theta_int})


    ###
    # Generate data from exact ODE and generate observations
    ###

    # set up required variables
    simulated_data = np.zeros([len(env), L*d])
    simulated_model = np.array([None]+[[]]*n_env)[1:]
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
        if silent==False:
            print("Currently solving ODE-system on environment "+str(i))

        # catch warnings from ODE solver --> not a good model
        simulated_model[i] = scipy.integrate.solve_ivp(y0 = initial, 
                                                       fun = lambda t,x:reactions(t,x,theta), 
                                                       t_span=[0,100], 
                                                       t_eval=time_grid, 
                                                       method=ode_solver
                                                      ).y.T
        if (not simulated_model[i].dtype==float) or (simulated_model[i].shape[0] < len(time_grid)):
            if not silent==True:
                print("Problem in ode-solver")
            return(None)

        # select time points and add noise
        if only_target_noise==True:
            tmp = simulated_model[i][time_index,:]

            if relativ==True:
                noise_var = tmp[:,target_ind].max() - tmp[:,target_ind].min() + 0.0000001 # diff of vector,matrix?
                noiseterm = np.random.randn(L*env_size) * noise_var
                noiseterm = noiseterm.reshape([env_size, L])
            else:
                noiseterm = np.random.randn(L*env_size) * noise_target
                noiseterm = noiseterm.reshape([env_size, L])

            simulated_data[env==i, :] = np.array([list(tmp.T.flatten())]*env_size)
            simulated_data[env==i, noise_target*L:(noise_target+1)*L]\
                = simulated_data[env==i, noise_target*L:(noise_target+1)*L] + noiseterm

        else:
            tmp = simulated_model[i][time_index,:]
            if relativ==True:
                noise_var = tmp.max(axis=0) - tmp.min(axis=0) + 0.0000001 # diff of vector,matrix?
                noiseterm = np.random.randn(L*d*env_size) * noise_var
                noiseterm = noiseterm.reshape([env_size, L*d])

            else:
                noiseterm = np.random.randn(L*d*env_size) * np.array(list(noise_var)*L*env_size)
                noiseterm = noiseterm.reshape([env_size, L*d])

            simulated_data[env==i,:] =  np.array([list(tmp)]*env_size) + noiseterm

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
        return(blowup)

    return({"simulated_data":simulated_data,
            "time":time,
            "env":env,
            "simulated_model":simulated_model,
            "true_model":true_set,
            "target":target})