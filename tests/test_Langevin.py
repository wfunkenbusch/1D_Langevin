#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import numpy as np
import Langevin
from Langevin import Langevin
from Langevin.Langevin import *

class RGK_unit_tests(unittest.TestCase):
    def test_increment(self):
        t = [0, 1]
        x0 = 0
        x = RGK(lambda x, t: t*x**2, t, [x0], rand = None)

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
        val1 = RGK(lambda x, t: t + x, t, [x0], rand = None)
        val2 = RGK(lambda x, t: t + x, t, [x0], rand = [0, 0, 1])
        self.assertNotEqual(val1[0, -1], val2[0, -1])

if __name__ == "__main__":
	unittest.main()
