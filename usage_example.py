import numpy as np
from matplotlib import pyplot as plt
from CausalKinetiX import CausalKinetiX
from CausalKinetiX.CausalKinetiX_modelranking import CausalKinetiX_modelranking
from CausalKinetiX.constrained_smoothspline import constrained_smoothspline
from CausalKinetiX.generate_data_maillard import generate_data_maillard
from CausalKinetiX.ode_solver import ode_solver

### This script contains some examples that show how to use the functions

###
# Example 1: Using CausalKinetiX
###

## Generate data from Maillard reaction
simulation_obj = generate_data_maillard(target=5,
                                        env=np.repeat([0, 1, 2], 3),
                                        L=15,
                                        noise_sd=1)

D = simulation_obj['simulated_data']
time = simulation_obj['time']
env = simulation_obj['env']
target = simulation_obj['target']

## Fit data using CausalKinetiX
ck_fit = CausalKinetiX(D, time, env, target,
                       pars={'expsize': 1,
                             'average_reps': True,
                             'show_plot': False,
                             'silent': True})

# variable ranking (here the true parent is variable 3 (i.e. X^4))
print(ck_fit['ranking'])


###
# Example 2: Using CausalKinetiX_modelranking
###

## Generate data from Maillard reaction
simulation_obj = generate_data_maillard(target=0,
                                        env=np.repeat([0, 1, 2, 3, 4], 3),
                                        L=20,
                                        noise_sd=1)
D = simulation_obj['simulated_data']
time = simulation_obj['time']
env = simulation_obj['env']
target = simulation_obj['target']
## Fit data to the following two models using CausalKinetiX:
## 1: dy = theta_1*x_1 + theta_2*x_2 + theta_3*x_1*x_10 (true model)
## 2: dy = theta_1*x_2 + theta_2*x_4 + theta_3*x_3*x_10 (wrong model)
ck_fit = CausalKinetiX_modelranking(D, time, env, target,
                                    models=[[[0], [1], [0, 9]],
                                            [[1], [3], [2, 9]]])
print(ck_fit)


###
# Example 3: Using constrained_smoothspline
###

x = np.arange(0, 4, 4/200)
x_long = np.arange(0, 4, 4/1000)
y = x**2 + np.random.normal(0, 2, 200)
dy = 2*x
dybdd = np.hstack([dy-0.5, dy+0.5])

fit = constrained_smoothspline(y=y,
                               times=x,
                               pen_degree=2,
                               constraint="none",
                               derivative_values=None,
                               times_new=x_long,
                               num_folds=5,
                               lambd="optim")

plt.plot(x, y, 'o')
plt.plot(x_long, fit['smooth_vals_new'], '-')
plt.show()


###
# Example 4: Using ode_solver
###

## Generate data from Maillard reaction
simulation_obj = generate_data_maillard(target=4,
                                        env=np.repeat(0, 5),
                                        L=20,
                                        noise_sd=0.1,
                                        only_target_noise=False,
                                        relativ=True)

D = simulation_obj['simulated_data']
time = simulation_obj['time']
env = simulation_obj['env']
target = simulation_obj['target']
L = len(time)
times = np.tile(time, 5)
X = np.empty([5*L, 11])
for j in range(11):
    X[:, j] = D[:, (L*j):(L*(j+1))].reshape([L*5])
## Solve for Melanoidin (X^11)
odefit = ode_solver(time, np.array([0]), times, X, [[7]], 10, np.array([0.12514]),
                    included_vars=np.array([7]))

plt.plot(odefit[:, 0], odefit[:, 1])
plt.plot(times, X[:, 10], 'o')
plt.show()
