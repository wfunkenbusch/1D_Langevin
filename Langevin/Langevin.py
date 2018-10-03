# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def RGK(fun, t, y0, vals, wall_size, rand = None):
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

    wall_size (float):
    The location of the second wall. The first wall is placed at x = 0. If the particle hits the wall, the simulation will stop.

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
        if F[0, i] <= 0 or F[0, i] >= wall_size:
            F = F[:, :i + 1]
            t = t[:i + 1]
            break
        yn = F[:, i] #sets the current values in yn
        tn = t[i] #sets the current time
        h = t[i + 1] - t[i] #sets the current time step

	#RGK parameters
        k1 = h*np.array(fun(tn, yn, vals))
        k2 = h*np.array(fun(tn + h/2, yn + k1/2, vals))
        k3 = h*np.array(fun(tn + h/2, yn + k2/2, vals))
        k4 = h*np.array(fun(tn + h, yn + k3, vals))

        F[:, i + 1] = yn + 1/6*(k1 + 2*k2 + 2*k3 + k4) #sets the next values in F
        if rand != None:
            F[rand[0], i + 1] += np.random.normal(loc = rand[1], scale = rand[2])*h/vals[0] #adds a random aspect to the values, if desired
	
    return t, F

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
    Takes Brownian motion parameters and converts them into the format to be used in RGK. All parameters are in reduced units.
    
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
    A scaling parameter for the standard deviation of the random force. Default 1e-20.
	
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
        rand = [1, 0, np.sqrt(2*1*T*Lambda*dt)] #Adds to velocity, centered at 0, standard deviation of sqrt(2k_B*T*lambda*(t - t'))
                                                #Note that kB = 1 in reduced units
    vals = [m, gamma] #mass and damping coefficient
    x0 = [init_pos, init_vel] #initial position and velocity
    	
    return t, rand, vals, x0

def Langevin(t_t, dt, init_pos, init_vel, m, gamma, T, wall_size, Lambda = 1e-20, rand = 'yes'):
    '''
    Takes Brownian motion parameters and outputs the time, position, and velocity arrays.
    
    Arguments:
    See RGK
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

    t, ans = RGK(ODE, t, x0, vals, wall_size, rand = rand)
    x = ans[0, :]
    v = ans[1, :]

    return t, x, v

def Save(FileName, t, x, v, p = 'No'):
    '''
    Takes a file name, times, positions, and velocities, and saves them in a file.
    
    Arguments:
    FileName (string):
    The name of the file to be saved (with file extension)

    print:
    If 'No', does not print final position and velocity. Will still save values to a file. Default 'No'.
    
    t, x, v (arrays):
    See Langevin.

    Saves:
    File with indices, times, positions, and velocities.

    Returns:
    x[-1], v[-1] (floats):
    The final position and velocity, respectively, of the particle.
    '''
	
    #prints the final position and velocity

    lines = ['index, t, x, v\n'] #list to store the lines to be saved into the file
       			                 #the first line has headings for each column

    #stores each set of values
    for i in range(len(x)):
        lines.append(str(i) + ' ' + str(t[i]) + ' ' + str(x[i]) + ' ' + str(v[i]) + '\n')
	
    #saves lines into a file
    F = open(FileName, 'w')
    for i in range(len(lines)):
        F.write(lines[i])
    F.close()

    if p != 'No':
        print('The final particle position is {}' .format(x[-1]))
        print('The final particle velocity is {}' .format(v[-1]))
    return x[-1], v[-1]

def Hist(FileName, t_t, dt, init_pos, init_vel, m, gamma, T, wall_size, Lambda = 1, rand = 'yes', trials = 100, p = 'No', s = 'No'):
    '''
    Takes a file name and Brownian motion parameters and outputs a histogram with the amount of time it took to hit a wall. Also saves the data in files.

    Arguments:
    FileName (string):
    The base name of the file to be saved. All trials will be saved under the format FileName_i.txt where i is the trial number (indexed from 0), the histogram will be saved as FileName.pdf, and the times will be saved as FileName_times.txt

    trials (float):
    The number of trials to be performed. Default 100.

    s:
    If 'No', does not save the data individual data files for the histogram.

    See RGK, Save, and params for other arguments.

    Saves:
    Text files for each trial containing the indices, times, positions, and velocities. Saved as FileName_i.txt where i is the trial number (indexed from 0)

    A histogram containing the amount of time for each trial to reach either wall (see wall_size). Saved as FileName.pdf
    '''
    
    times = [] #will store the times to be saved in the histogram
    for i in range(trials):
        t, x, v = Langevin(t_t, dt, init_pos, init_vel, m, gamma, T, wall_size, Lambda, rand = 'yes') #runs a simulation
        if s != 'No':
            xf, vf = Save(FileName + '_' + str(i) + '.txt', t, x, v, p)
        if x[-1] <= 0 or x[-1] >= wall_size: #only add time if particle hit a wall
            times.append(t[-1])
    
    #plotting
    f = plt.figure()
    plt.xlabel('Time', fontsize = 16)
    plt.ylabel('Frequency', fontsize = 16)
    plt.hist(times, bins = 'auto')
    f.savefig(FileName + '_hist.pdf', bbox_inches = 'tight')

def Plot(FileName, t_t, dt, init_pos, init_vel, m, gamma, T, wall_size, Lambda = 1, rand = 'yes', p = 'No'):
    '''
    Plots position vs. time for Brownian motion.

    Arguments:
    FileName (string):
    The base name of the file which will contain the graph. Will be saved as a pdf file.

    See Hist for other arguments

    Saves:
    A plot of position vs. time for a single simulation of Brownian motion.
    '''

    t, x, v = Langevin(t_t, dt, init_pos, init_vel, m, gamma, T, wall_size, Lambda, rand = 'yes') #runs a simulation
    Save(FileName + '_plot.txt', t, x, v, p)
    f = plt.figure()
    plt.xlabel('Time', fontsize = 16)
    plt.ylabel('Position', fontsize = 16)
    plt.plot(t, x)
    f.savefig(FileName + '_plot.pdf', bbox_inches = 'tight')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--FileName', type = str, default = 'd', help = 'String: Base file name')
    parser.add_argument('--t_t', type = float, default = 1000, help = 'Float: Total time of simulation')
    parser.add_argument('--dt', type = float, default = 1e-1, help = 'Float: Time step of simulation')
    parser.add_argument('--init_pos', type = float, default = 2.5, help = 'Float: Initial position of particle')
    parser.add_argument('--init_vel', type = float, default = 0, help = 'Float: Initial velocity of particle')
    parser.add_argument('--m', type = float, default = 1, help = 'Float: Mass of particle')
    parser.add_argument('--gamma', type = float, default = 1, help = 'Float: Damping coefficient')
    parser.add_argument('--T', type = float, default = 300, help = 'Float: Temperature')
    parser.add_argument('--Lambda', type = float, default = 1, help = 'Float: Variance parameter for random force')
    parser.add_argument('--trials', type = int, default = 1000, help = 'Integer: Number of trials to run')
    parser.add_argument('--wall_size', type = float, default = 5, help = 'Float: Position of second wall (first wall at 0)')
    parser.add_argument('--rand', type = str, default = 'Yes', help = 'String: Whether to apply the random force, None if no random force')
    parser.add_argument('--p', type = str, default = 'Yes', help = 'String: Whether to print the final result, "No" if no printing')
    parser.add_argument('--s', type = str, default = 'No', help = 'Whether to save histogram data files, "No" if no saveing')
    
    args, unknown = parser.parse_known_args()

    return args
    
def main():
    args = get_parser()
    Plot(args.FileName, args.t_t, args.dt, args.init_pos, args.init_vel, args.m, args.gamma, args.T, args.wall_size, args.Lambda, args.rand, args.p)
    Hist(args.FileName, args.t_t, args.dt, args.init_pos, args.init_vel, args.m, args.gamma, args.T, args.wall_size, args.Lambda, args.rand, args.trials, args.p, args.s)

if __name__ == '__main__':
    main()