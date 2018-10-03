# 1D_Langevin

[![Coverage Status](https://coveralls.io/repos/github/wfunkenbusch/1D_Langevin/badge.svg?branch=master)](https://coveralls.io/github/wfunkenbusch/1D_Langevin?branch=master)

Simulates 1D Brownian motion using Langevin Dynamics for a system without potential energy. Uses Runge-Kutta numerical integration. Plots the path of a single simulation and a histogram of the amount of time it takes to reach a desired point.

Implementation:
1. Must be run on Python 3.5 or above
1. Needed outside modules: numpy, matplotlib, argparse, os (if unit testing)
2. Clone this repository using the following command: git clone https://github.com/wfunkenbusch/1D_Langevin.git
3. Enter the base directory (cd 1D_Langevin)
4. To run the code, use the following command: python Langevin/Langevin.py --arguments

Arguments:

All arguments are in reduced units, and may be input in any order. If no value is specified for a particular input, the default is used.

Ex.

python Langevin/Langevin.py --FileName example --dt 1e-2 --init_pos 5 --wall_size 10 --init_vel 1

* --FileName
    * type: string
    * default: 'd'
    * The base file name for the output files. Histogram will be saved as FileName_hist.pdf. Plot will be saved as FileName_plot.pdf. Plot data will be saved as FileName_plot.txt. Histogram data files (optional) will be saved as FileName_RunNumber.txt.

* --t_t
    * type: float
    * default: 1000
    * The total time of the simulation. The simulation will be run for t_t/dt time steps, rounded down.

* --dt
    * type: float
    * default: 1e-1
    * The time step of the simulation. The simulation will be run for t_t/dt time steps, rounded down.

* --init_pos
    * type: float
    * default: 2.5
    * The initial position of the particle. If less than 0 or greater than wall_size, the current simulation will exit immediately.

* --init_vel
    * type: float
    * default: 0
    * The initial velocity of the particle.

* --m
    * type: float
    * default: 1
    * The mass of the particle.

* --gamma
    * type: float
    * default: 1
    * The damping coefficient on the particle. Represents the drag force on the particle.

* --T
    * type: float
    * default: 300
    * The temperature of the system, in K.

* --Lambda
    * type: float
    * default: 1
    * A scaling factor for the variance in the random force on the particle.

* --trials
    * type: integer
    * default: 1000
    * The number of trials to run for the histogram. An additional trial is always run for the plot.

* --wall_size
    * type: float
    * default: 5
    * The location of one of the walls in the system. The other wall is at x = 0. If the particle hits either wall, the current simulation will stop.

* --rand
    * type: any
    * default: 'Yes'
    * For testing purposes. An option for removing the random force on the particle. Must be set to None to remove the random force.

* --p
    * type: str
    * default: 'Yes'
    * Decides whether to print the final position and velocity of the particle in the plot simulation. Must be set to 'No' to not print the results.

* --s
    * type: str
    * default: 'No'
    * Decides whether to save the histogram data files. If set to anything other than 'No', will save files. CAUTION: for a large number of trials or simulations with many time steps, this may require a lot of storage.

Outputs:

Data files are stored as .txt files. They are formated as *index time position velocity* with labels at the top and each index at a new line.

Graphs are stored as .pdf files.

* FileName_plot.pdf:
    * A plot of position vs. time for a single run. The x-axis is automatically labeled as "Time", and the y-axis as "Position."

* FileName_plot.txt:
    * The plot's data file

* FileName_hist.pdf:
    * A histogram of the amount of time to hit the wall. Trials which do not hit the wall are not recorded.

* FileName_RunNumber.txt:
    * Optional (see --s). RunNumber starts from 0. Stores each individual run's data for the histogram trials.
