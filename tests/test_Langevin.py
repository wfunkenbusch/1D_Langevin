#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import numpy as np
import Langevin
from Langevin import Langevin
from Langevin.Langevin import *

class RGK_unit_tests(unittest.TestCase):
    def test(self):
        t = [0, 1]
        x0 = 0
        x = RGK(lambda x, t: t*x**2, t, [x0])

        h = t[1] - t[0]
        k1 = h*t[0]*x0**2
        k2 = h*(t[0] + h/2)*(x0 + k1/2)**2
        k3 = h*(t[0] + h/2)*(x0 + k2/2)**2
        k4 = h*(t[0] + h)*(x0 + k3)**2

        xf = x0 + 1/6*(k1 + 2*k2 + 2*k3 + k4)
        self.assertEqual(x[0, 1], xf)

if __name__ == "__main__":
	unittest.main()
