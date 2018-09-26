# -*- coding: utf-8 -*-

import numpy as np

def RGK(fun, t, y0, vals, rand = None):
    '''
    Takes a function, time range, and initial conditions, and numerically integratesthe function according to the Runge-Kutta Method.

    Arguments:
    fun (function):
    
    The ODE('s) to be integrated. Should take the independent variable (t) and the dependent variable (y) as arguments, in that order. Should return the derivative of y with respect to t.

    For multiple ordinary differential equations, the dependent variable input should be a list of each variable. The output should return each derivative in the same order that the variables were input, as a list.

    It is important that the initial conditions (y0) are in the same order as the output of fun.

    t (array):

    The array of values to be integrated over. t[0] is the initial condition.

    y0 (list or array):

    The initial values of each variable to be integrated. Must be in the same order as the output of fun.

    vals (list or array):

    The problem parameters which fun must take in addition to t and y. Should be in the same order that they are required in the function.

    rand (list or array):

    A random value to be added to a variable. The first input is the variable to be added to (indexed 0, 1,...). The second input is the mean of the distribution, the third input is the standard deviation of the distribution. The value is pulled from a normal distribution. The value will be multiplied by the change in t at that time step and divided by the mass, m (vals[0]). This option can be opted out of by setting rand = None, which is the default.

    Returns:
    F (matrix): 
    
    The solution to the ODE's stacked vertically. The order of stacking is the same as the order of the output of fun.
    '''
    F = np.zeros((len(y0), len(t)))
    F[:, 0] = y0

    yn = np.zeros(len(y0))

    for i in range(len(t) - 1):
        yn = F[:, i]
        tn = t[i + 1]
        h = t[i + 1] - t[i]

        k1 = h*np.array(fun(tn, yn, vals))
        k2 = h*np.array(fun(tn + h/2, yn + k1/2, vals))
        k3 = h*np.array(fun(tn + h/2, yn + k2/2, vals))
        k4 = h*np.array(fun(tn + h, yn + k3, vals))

        F[:, i + 1] = yn + 1/6*(k1 + 2*k2 + 2*k3 + k4)
        if rand != None:
            F[rand[0], i + 1] += np.random.normal(rand[1], rand[2])*h/vals[0]

    return F

def ODE(t, x0, vals):
    '''
    Takes a time, position, and velocity, and returns the velocity and acceleration for 1D Brownian motion with no random force.

    Arguments:

    t (float):

    The independent variable.

    x0 (list or array):

    The dependent variables. The first input should be the position, and the second input should be the velocity.

    vals (list or array):

    The Brownian motion parameters. The first input should be mass, the second input should be the damping coefficient (gamma) (all in SI units).

    Returns:

    [dxdt, dvdt] (list):

    The velocity and acceleration at the specified time, position, and velocity.
    '''
    x = x0[0]
    v = x0[1]
    m = vals[0]
    gamma = vals[1]
    
    dxdt = v
    dvdt = -gamma*v/m

    return [dxdt, dvdt]

def params(t_t, dt, init_pos, init_vel, m, gamma, T, Lambda, rand = 'yes'):
    t = np.linspace(0, t_t, int(t_t//dt + 1))
    if rand != None:
        rand = [1, 0, np.sqrt(2*1.38064852*10**(-23)*T*Lambda*dt)] #Adds to velocity, centered at 0, standard deviation of sqrt(2k_B*T*lambda*(t - t'))
    vals = [m, gamma]
    x0 = [init_pos, init_vel]
    	
    return t, rand, vals, x0

def Langevin(FileName, t_t, dt, init_pos, init_vel, m, gamma, T, Lambda = 1, rand = 'yes'):
    '''
    Takes a file name and Langevin parameters and simulates Brownian motion of a particle. Prints the final position and velocity of the particle and saves the index, time, position, and velocity of the particle at each time step in a file with the given file name.

    Arguments:

    FileName (string):

    The filename for the particle properties to be saved to. Requires an extension.

    t_t (float):

    The total time for the simulation to run in seconds.

    dt (float):

    The time step in seconds.

    init_pos (float):

    The initial position of the particle in m.

    init_vel (float):

    The initial velocity of the particle in m/s.

    m (float):

    The mass of the particle in kg.

    gamma (float):

    The damping coefficient in kg/s.

    T (float):

    The temperature of the system.

    Lambda (float):

    A scaling parameter for the standard deviation of the random force.

    rand (value):

    If None, the random function is disabled.

    Prints:

    x[-1] (float):

    The final position of the particle.

    v[-1] (float):

    The final velocity of the particle.

    Saves:

    The index, time, position and velocity of the particle at each time step.
    '''
    t, rand, vals, x0 = params(t_t, dt, init_pos, init_vel, m, gamma, T, Lambda)

    ans = RGK(ODE, t, x0, vals, rand = rand)
    x = ans[0, :]
    v = ans[1, :]

    print('The final position of the particle was {:.3} m.' .format(x[-1]))
    print('The final velocity of the particle was {:.3} m/s.' .format(v[-1]))

    lines = ['index, t, x, v']

    for i in range(len(x)):
        lines.append(str(i) + ' ' + str(t[i]) + ' ' + str(x[i]) + ' ' + str(v[i]) + '\n')

    F = open(FileName, 'w')
    for i in range(len(lines)):
        F.write(lines[i])
    F.close()

