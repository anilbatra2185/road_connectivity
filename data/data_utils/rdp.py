#!/usr/bin/env python3


"""
The Ramer-Douglas-Peucker algorithm roughly ported from the pseudo-code provided
by http://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm

The code is taken from
https://github.com/mitroadmaps/roadtracer/blob/master/lib/discoverlib/rdp.py
"""

from math import sqrt


def distance(a, b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def point_line_distance(point, start, end):
    """
    Calaculate the prependicuar distance of given point from the line having
    start and end points.
    """
    if start == end:
        return distance(point, start)
    else:
        n = abs(
            (end[0] - start[0]) * (start[1] - point[1])
            - (start[0] - point[0]) * (end[1] - start[1])
        )
        d = sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        return n / d


def rdp(points, epsilon):
    """
    Reduces a series of points to a simplified version that loses detail, but
    maintains the general shape of the series.

    @param points: Series of points for a line geometry represnted in graph.
    @param epsilon: Tolerance required for RDP algorithm to aproximate the
                    line geometry.

    @return: Aproximate series of points for approximate line geometry
    """
    dmax = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        d = point_line_distance(points[i], points[0], points[-1])
        if d > dmax:
            index = i
            dmax = d
    if dmax >= epsilon:
        results = rdp(points[: index + 1], epsilon)[:-1] + rdp(points[index:], epsilon)
    else:
        results = [points[0], points[-1]]
    return results
