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
    F = np.zeros((len(y0), len(t))) #array to store the final values of the dependent variables
    F[:, 0] = y0 #sets the initial values

    yn = np.zeros(len(y0)) #array to store the current values of the dependent variables

    for i in range(len(t) - 1):
        yn = F[:, i] #sets the current values in yn
        tn = t[i + 1] #sets the current time
        h = t[i + 1] - t[i] #sets the current time step

		#RGK parameters
        k1 = h*np.array(fun(tn, yn, vals))
        k2 = h*np.array(fun(tn + h/2, yn + k1/2, vals))
        k3 = h*np.array(fun(tn + h/2, yn + k2/2, vals))
        k4 = h*np.array(fun(tn + h, yn + k3, vals))

        F[:, i + 1] = yn + 1/6*(k1 + 2*k2 + 2*k3 + k4) #sets the next values in F
        if rand != None:
            F[rand[0], i + 1] += np.random.normal(rand[1], rand[2])*h/vals[0] #adds a random aspect to the values, if desired

    return F

def ODE(t, x0, vals):
    '''
    Takes a time, position, and velocity, and returns the velocity and acceleration for 1D Brownian motion with no random force.

    Arguments:
    t (float):
    The current time.

    x0 (list or array):
    The first input should be the position, and the second input should be the velocity.

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
    
    dxdt = v #position is the derivative of velocity
    dvdt = -gamma*v/m #Langevin value for acceleration

    return [dxdt, dvdt]

def params(t_t, dt, init_pos, init_vel, m, gamma, T, Lambda, rand = 'yes'):
    '''
    Takes Brownian motion parameters and converts them into the format to be used in RGK. All parameters are in SI units.
    
    Arguments:
    t_t (float):
    The total time for the simulation.
    
    dt (float):
    The time step for the simulation.
    
    init_pos (float):
    The initial position of the particle.
    
    init_vel (float):
    The initial velocity of the particle.
    
    m (float):
    The mass of the particle.
    
    gamma (float):
    The damping coefficient of the particle in the fluid.
    
    T (float):
    The temperature of the system.
    
    Lambda (float):
    A scaling parameter for the standard deviation of the random force.
	
    rand:
    If None, the random function is disabled.
	
    Returns:
    t (array):
    An array which contains the times.
    
    rand (array):
    An array which contains the random variable parameters.
    
    vals (array):
    An array which contains m in the first position and gamma in the second position.
    
    x0 (array):
    An array which contains init_pos in the first position and init_vel in the second position.
    '''
    t = np.linspace(0, t_t, int(t_t//dt + 1)) #starts at 0 and ends at t_t
    if rand != None:
        rand = [1, 0, np.sqrt(2*1.38064852*10**(-23)*T*Lambda*dt)] #Adds to velocity, centered at 0, standard deviation of sqrt(2k_B*T*lambda*(t - t'))
    vals = [m, gamma] #mass and damping coefficient
    x0 = [init_pos, init_vel] #initial position and velocity
    	
    return t, rand, vals, x0

def Langevin(t_t, dt, init_pos, init_vel, m, gamma, T, Lambda = 1, rand = 'yes'):
    '''
    Takes Brownian motion parameters and outputs the time, position, and velocity arrays.
    
    Arguments:
    See params
    
    Returns:
    t (array)
    An array which contains the times.
    
    x (array):
    An array which contains the positions.
    
    v (array):
    An array which contains the velocities.
    '''
    t, rand, vals, x0 = params(t_t, dt, init_pos, init_vel, m, gamma, T, Lambda)

    ans = RGK(ODE, t, x0, vals, rand = rand)
    x = ans[0, :]
    v = ans[1, :]

    return t, x, v

def Save(FileName, t, x, v):
    '''
    Takes a file name, times, positions, and velocities, and saves them in a file.
    
    Arguments:
    FileName (string):
    The name of the file to be saved (with file extension)
    
    t, x, v (arrays):
    See Langevin.
    '''
	
    #prints the final position and velocity
    print('The final position of the particle was {} m.' .format(x[-1]))
    print('The final velocity of the particle was {} m/s.' .format(v[-1]))

    lines = ['index, t, x, v'] #list to store the lines to be saved into the file
       			       #the first line has headings for each column

    #stores each set of values
    for i in range(len(x)):
        lines.append(str(i) + ' ' + str(t[i]) + ' ' + str(x[i]) + ' ' + str(v[i]) + '\n')
	
    #saves lines into a file
    F = open(FileName, 'w')
    for i in range(len(lines)):
        F.write(lines[i])
    F.close()

