#!/usr/bin/env python3


import math

import numpy as np
import data_utils.rdp as rdp


def simplify_edge(ps, max_distance=1):
    """
    Combine multiple points of graph edges to line segments
    so distance from points to segments <= max_distance

    @param ps: array of points in the edge, including node coordinates
    @param max_distance: maximum distance, if exceeded new segment started

    @return: ndarray of new nodes coordinates
    """
    res_points = []
    cur_idx = 0
    for i in range(1, len(ps) - 1):
        segment = ps[cur_idx : i + 1, :] - ps[cur_idx, :]
        angle = -math.atan2(segment[-1, 1], segment[-1, 0])
        ca = math.cos(angle)
        sa = math.sin(angle)
        # rotate all the points so line is alongside first column coordinate
        # and the second col coordinate means the distance to the line
        segment_rotated = np.array([[ca, -sa], [sa, ca]]).dot(segment.T)
        distance = np.max(np.abs(segment_rotated[1, :]))
        if distance > max_distance:
            res_points.append(ps[cur_idx, :])
            cur_idx = i
    if len(res_points) == 0:
        res_points.append(ps[0, :])
    res_points.append(ps[-1, :])

    return np.array(res_points)


def simplify_graph(graph, max_distance=1):
    """
    @params graph: MultiGraph object of networkx
    @return: simplified graph after applying RDP algorithm.
    """
    all_segments = []
    # Iterate over Graph Edges
    for (s, e) in graph.edges():
        for _, val in graph[s][e].items():
            # get all pixel points i.e. (x,y) between the edge
            ps = val["pts"]
            # create a full segment
            full_segments = np.row_stack(
                [graph.nodes[s]["o"], ps, graph.nodes[e]["o"]])
            # simply the graph.
            segments = rdp.rdp(full_segments.tolist(), max_distance)
            all_segments.append(segments)

    return all_segments


def segment_to_linestring(segment):
    """
    Convert Graph segment to LineString require to calculate the APLS mteric
    using utility tool provided by Spacenet.
    """

    if len(segment) < 2:
        return []
    linestring = "LINESTRING ({})"
    sublinestring = ""
    for i, node in enumerate(segment):
        if i == 0:
            sublinestring = sublinestring + "{:.1f} {:.1f}".format(node[1], node[0])
        else:
            if node[0] == segment[i - 1][0] and node[1] == segment[i - 1][1]:
                if len(segment) == 2:
                    return []
                continue
            if i > 1 and node[0] == segment[i - 2][0] and node[1] == segment[i - 2][1]:
                continue
            sublinestring = sublinestring + ", {:.1f} {:.1f}".format(node[1], node[0])
    linestring = linestring.format(sublinestring)
    return linestring


def segmets_to_linestrings(segments):
    """
    Convert multiple segments to LineStrings require to calculate the APLS mteric
    using utility tool provided by Spacenet.
    """

    linestrings = []
    for segment in segments:
        linestring = segment_to_linestring(segment)
        if len(linestring) > 0:
            linestrings.append(linestring)
    if len(linestrings) == 0:
        linestrings = ["LINESTRING EMPTY"]
    return linestrings


def unique(list1):
    # intilize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list
