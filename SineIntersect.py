#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 15:03:25 2022

@author: srilthe

use sine wave to calculate 'bounce' points in mirrored walls
"""

import numpy as np
import matplotlib.pyplot as plt


def gen(ar):
    box = []
    arlen = len(ar)
    for i in range(arlen):
        box.append(ar[i])
        box.append(ar[(i+1) % arlen])
    lx, ly = zip(*box)
    return(lx, ly)


x1 = -1000
x2 = 1000
y1 = -1000
y2 = 1000

p1 = (33, 66)
p2 = (155, 210)

#  plt.axis([x1, x2, y1, y2])
plt.axis('on')
plt.grid(True)

box = [[0,0], [0, 250], [250, 250], [250, 0]]
xs, ys = gen(box)
plt.plot(ys, xs, 'b')
plt.plot(p1[0], p1[1], 'go')
plt.plot(p2[0], p2[1], 'ro')


#  one bounce
p3 = (p1[0], p2[1])
plt.plot(p3[0], p3[1], 'bo')
#  midpoint of p2 & p3 is a bounce
#  project that onto top wall


plt.show()
