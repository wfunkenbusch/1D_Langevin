#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import numpy as np
import os.path
import Langevin
from Langevin import Langevin
from Langevin.Langevin import *

class RGK_unit_tests(unittest.TestCase):
    def test_increment(self):
        t = [0, 1]
        x0 = 0
        vals = [1, 1]
        x = RGK(lambda x, t, vals: t*x**2, t, [x0], vals, rand = None)

        h = t[1] - t[0]
        k1 = h*t[0]*x0**2
        k2 = h*(t[0] + h/2)*(x0 + k1/2)**2
        k3 = h*(t[0] + h/2)*(x0 + k2/2)**2
        k4 = h*(t[0] + h)*(x0 + k3)**2

        xf = x0 + 1/6*(k1 + 2*k2 + 2*k3 + k4)
        self.assertEqual(x[0, 1], xf)

    def test_rand(self):
        np.random.seed(1234)
        t = np.linspace(0, 1, 100)
        x0 = 0
        vals = [1, 1, 1]
        val1 = RGK(lambda x, t, vals: t + x, t, [x0], vals, rand = None)
        val2 = RGK(lambda x, t, vals: t + x, t, [x0], vals, rand = [0, 0, 1])
        self.assertNotEqual(val1[0, -1], val2[0, -1])

    def test_double_pos(self):
        t = [0, 1]
        x0 = [1, 1]
        vals = [1, 1]

        def fun(t, x0, vals):
            return [1, 1]

        x = RGK(fun, t, x0, vals, rand = None)
        xf = x[0, 1]

        self.assertEqual(xf, 2)

    def test_double_vel(self):
        t = [0, 1]
        x0 = [1, 1]
        vals = [1, 1]
        
        def fun(t, x0, vals):
            return [1, 1]

        x = RGK(fun, t, x0, vals, rand = None)
        vf = x[1, 1]

        self.assertEqual(vf, 2)

class ODE_unit_tests(unittest.TestCase):
    def test_position(self):
        dxdt, dvdt = ODE(0, [1, 1], [1, 1])
        self.assertEqual(dxdt, 1)

    def test_velocity(self):
        dxdt, dvdt = ODE(0, [1, 1], [1, 1])
        self.assertEqual(dvdt, -1)

    def test_stationary(self):
        dxdt, dvdt = ODE(0, [1, 0], [1, 1])
        self.assertEqual(dxdt, 0)

    def test_no_drag(self):
        dxdt, dvdt = ODE(0, [1, 1], [1, 0])
        self.assertEqual(dvdt, 0)

class params_unit_tests(unittest.TestCase):
    def test_params(self):
        t, rand, vals, x0 = params(1, 0.5, 3, 5, 7, 11, 13, 17)
        self.assertEqual(t[1], 0.5)
        self.assertEqual(rand[2], np.sqrt(2*1.38064852*10**(-23)*13*17*0.5))
        self.assertEqual(vals[0], 7)
        self.assertEqual(vals[1], 11)
        self.assertEqual(x0[0], 3)
        self.assertEqual(x0[1], 5)
		
class Langevin_unit_tests(unittest.TestCase):
    def test_file(self):
        np.random.seed(1234)
        Langevin(FileName = 'file_test.txt', t_t = 1, dt = 1e-3, init_pos = 0, init_vel = 0, m = 1e-9, gamma = 1e-10, T = 300, Lambda = 1)
        self.assertTrue(os.path.exists('file_test.txt'))

    def test_output(self):
        F = open('file_test.txt', 'r')
        last_line = F.readlines()[-1]
        F.close()
        self.assertEqual('999 1.0 3.828046861000748e-05 4.3831785467278256e-05\n', last_line)

if __name__ == "__main__":
    unittest.main()
