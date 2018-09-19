# -*- coding: utf-8 -*-

import numpy as np

def RGK(fun, t, y0):
    '''
    Takes a function, time range, and initial conditions, and numerically integratesthe function according to the Runge-Kutta Method.

    Arguments:
    fun (function):
    
    The ODE('s) to be integrated. Should take the independent variable (t) and the dependent variable (y) as arguments, in that order. Should return the derivative ofy with respect to t.

    For multiple ordinary differential equations, the dependent variable input should be a list of each variable. The output should return each derivative in the same order that the variables were input.

    It is important that the initial conditions (y0) are in the same order as the output of fun.

    t (array):

    The array of values to be integrated over. t[0] is the initial condition.

    y0 (list or array):

    The initial values of each variable to be integrated. Must be in the same order as the output of fun.
    '''
    F = np.zeros((len(y0), len(t)))
    F[:, 0] = y0

    yn = np.zeros(len(y0))

    for i in range(len(t) - 1):
        yn = F[:, i]
        tn = t[i + 1]
        h = t[i + 1] - t[i]

        k1 = h*fun(tn, yn)
        k2 = h*fun(tn + h/2, yn + k1/2)
        k3 = h*fun(tn + h/2, yn + k2/2)
        k4 = h*fun(tn + h, yn + k3)

        F[:, i + 1] = yn + 1/6*(k1 + 2*k2 + 2*k3 + k4)

    return F
        
