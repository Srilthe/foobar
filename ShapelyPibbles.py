#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 20:05:49 2022

@author: srilthe
"""


from shapely.geometry import box, LineString, Point, Polygon
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np
import math


def seg_intersect(ar):
    """
    Calculates where two lines would meet
    seg_intersect((x, y), (x, y), (x, y), (x, y))
    """
    line1 = LineString([(list(ar[0])),(list(ar[1]))])
    line2 = LineString([(list(ar[2])),(list(ar[3]))])
    geoms = line1.intersection(line2)
    arr = np.array(geoms.coords)
    if len(arr):
        return(arr)


def mirror_x(anchor, second):
    """
    Returns x, y co-ordinates with second mirrored in x plane
    """
    ax, ay = anchor
    sx, sy = second
    flip = ay + ay - sy
    return([sx, flip])


def mirror_y(anchor, second):
    """
    Returns x, y co-ordinates with second mirrored in y plane
    """
    ax, ay = anchor
    sx, sy = second
    flip = ax + ax - sx
    return([flip, sy])


#  setup wall from dim
polygon = ([(0,0),(0,300),(300,300),(300,0)])
dim_x, dim_y = [300, 300]
west = [0, 0], [0, dim_y]
north = [0, dim_y], [dim_x, dim_y]
east = [dim_x, dim_y], [dim_x, 0]
south = [dim_x, 0], [0, 0]
room = (west[0], west[1], north[1], east[1], south[1])
roomy, roomx = zip(*room)

#  test these two points
p1 = np.array([250, 90])
p2 = np.array([80, 190])

points = []

#  all lines appended here
lines = []
two_lines = np.array(((p1), (p2 * [1, -1]), south[0], south[1]))
#  print("two lines", two_lines)
result = list(seg_intersect(two_lines)[0])
points.append(result)
lines.append(((p2[0], result[0]), (p2[1], result[1])))
lines.append(((p1[0], result[0]), (p1[1], result[1])))

#  new point
#  result = [195.35714285714286, 0.0]
#  projected up onto 'north' through p1 & p2
mult = dim_y / p1[1]
run = (p1[0] - result[0]) * mult
rise = (p1[1] - result[1]) * mult
new_point1 = [run + p1[0], rise + p1[1]]

mult = dim_y / p2[1]
run = (p2[0] - result[0]) * mult
rise = (p2[1] - result[1]) * mult
new_point2 = [run + p2[0], rise + p2[1]]

left, right = min(new_point1[0], new_point2[0]), max(new_point1[0], new_point2[0])
arr1 = [(new_point1), (result), (left, dim_y), (right, dim_y)]
intersect1 = list(seg_intersect(arr1)[0][:])
points.append(intersect1)

arr2 = [(new_point2), (result), (left, dim_y), (right, dim_y)]
intersect2 = list(seg_intersect(arr2)[0][:])
points.append(intersect2)

#  TEST sine wave to get bounce co-ords
Fs = 300
f = (np.arange(10) / 2)[1:]
sample = 300
t = x = np.arange(sample)
i = 4.5
y = np.sin(2 * np.pi * i * x / Fs) * 150
series = np.sin(2 * np.pi * i * x / Fs) * 150
thresh = 0.95
peak_idx, _ = find_peaks(series, height=thresh)
valley_idx, _ = find_peaks(-series, height=thresh)


#  zip all co-ordinates
"""

"""
roomy, roomx = zip(*room)
if len(points):
    pointsy, pointsx = zip(*points)

# just plt below here
plt.figure()
plt.style.use('seaborn-dark')
plt.plot(p1[0], p1[1], 'go')
plt.plot(p2[0], p2[1], 'bo')

plt.plot(roomy, roomx)
if len(points):
    plt.plot(pointsy, pointsx, 'co')

for i in lines:
    plt.plot(i[0], i[1], 'r')

plt.plot(t, series)

# Plot threshold
plt.plot([min(t), max(t)], [thresh, thresh], '--')
plt.plot([min(t), max(t)], [-thresh, -thresh], '--')

# Plot peaks (red) and valleys (blue)
plt.plot(t[peak_idx], series[peak_idx], 'r.')
plt.plot(t[valley_idx], series[valley_idx], 'b.')

plt.show()

"""
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt


Fs = 300
f = (np.arange(10) / 2)[1:]
color = list('bgrcmyk')

sample = 300
t = x = np.arange(sample)
i = 4.5
y = np.sin(2 * np.pi * i * x / Fs) * 150
# Input signal

series = np.sin(2 * np.pi * i * x / Fs) * 150 

# Threshold value (for height of peaks and valleys)
thresh = 0.95

# Find indices of peaks
peak_idx, _ = find_peaks(series, height=thresh)

# Find indices of valleys (from inverting the signal)
valley_idx, _ = find_peaks(-series, height=thresh)

# Plot signal
plt.plot(t, series)

# Plot threshold
plt.plot([min(t), max(t)], [thresh, thresh], '--')
plt.plot([min(t), max(t)], [-thresh, -thresh], '--')

# Plot peaks (red) and valleys (blue)
plt.plot(t[peak_idx], series[peak_idx], 'r.')
plt.plot(t[valley_idx], series[valley_idx], 'b.')

plt.show()







"""
