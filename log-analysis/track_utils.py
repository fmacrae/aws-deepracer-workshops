import numpy as np
import matplotlib as mpl

# Shapely Library
from shapely.geometry import Polygon
from shapely.geometry.polygon import LineString
from typing import Tuple




def get_track_waypoints(track_name: str, absolute_path=".") -> np.array:
    return np.load("%s/tracks/%s.npy" % (absolute_path, track_name))
get_track_waypoints.__doc__ = """Returns a numpy array containing the waypoints for 
the inner, mid and outer track boundaries way points 
as a numpy array with the format 
[0:2]=center, [2:4]=inner and [4:6]=outer.  
First and last index are duplicates.  Use 
load_track as it splits the waypoints better."""

def load_track(track_name: str, absolute_path=".") -> Tuple[np.array, np.array, np.array, Polygon]:
    if track_name.endswith('.npy'):
        track_name = track_name[:-4]

    waypoints = get_track_waypoints(track_name, absolute_path)

    print("Loaded %s waypoints" % waypoints.shape[0])

    l_inner_border = LineString(waypoints[:, 2:4])
    l_outer_border = LineString(waypoints[:, 4:6])
    road_poly = Polygon(np.vstack((l_outer_border, np.flipud(l_inner_border))))

    # rescale waypoints to centimeter scale
    center_line = waypoints[:, 0:2] * 100
    inner_border = waypoints[:, 2:4] * 100
    outer_border = waypoints[:, 4:6] * 100

    return center_line, inner_border, outer_border, road_poly
load_track.__doc__ = """Returns 3 numpy arrays containing the coordinate tuples 
for the center, inner, outer and outer track boundaries 
way points and a Polygon containing the full road layout.  
Coordinates are scaled as centemeters.  
First and last index are duplicates."""

def get_angle(p0: np.array, p1: np.array, p2: np.array) -> np.degrees:
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
    return np.degrees(angle)
get_angle.__doc__ = """Takes three points x,y coordinates as input and returns the angle 
                       between at p1.  180 or -180 is a straight line."""


def get_vector_length(v: np.array) -> float:
    return np.sqrt(v[0] ** 2 + v[1] ** 2)
get_vector_length.__doc__ = """Returns the length of the Hypotenuse
for a vector passed in as a two row array
"""

def normalize_vector(v:np.array) -> np.array:
    return v / get_vector_length(v)
normalize_vector.__doc__ = """Convert vector to length of 1
for a vector passed in as a two row array
"""

# TODO Watch this, it might be putting points inside out
def get_orthogonal_vector_for_straight_line(before, point):
    v0 = np.array(before) - np.array(point)
    if v0[0] * v0[1] == 0:
        border_vector = -np.array([-v0[1], v0[0]])
    else:
        border_vector = -1 / np.array([-v0[0], v0[1]])

    border_vector = normalize_vector(border_vector)

    return border_vector


def crossing_point_for_two_lines(l1_p1, l1_p2, p, p2):
    if l1_p1[0] == l1_p2[0]:
        # if the waypoint line is vertical
        x = l1_p1[0]
        a1, b1 = get_a_and_b_for_line(p, p2)
    elif p[0] == p2[0]:
        # if the location line is vertical
        x = p[0]
        a1, b1 = get_a_and_b_for_line(l1_p1, l1_p2)
    else:
        a1, b1 = get_a_and_b_for_line(l1_p1, l1_p2)
        a2, b2 = get_a_and_b_for_line(p, p2)
        x = (b2 - b1) / (a1 - a2)
    y = a1 * x + b1
    return x, y


def get_a_and_b_for_line(p1, p2):
    a1 = (p1[1] - p2[1]) / (p1[0] - p2[0])
    b1 = p2[1] - a1 * p2[0]
    return a1, b1


def get_a_point_on_a_line_closest_to_point(l1_p1, l1_p2, p):
    vector = get_orthogonal_vector_for_straight_line((l1_p1[0], l1_p1[1]),
                                                     (l1_p2[0], l1_p2[1]))
    p2 = np.array([p[0], p[1]]) + vector
    crossing_point = crossing_point_for_two_lines(l1_p1, l1_p2, p, p2)
    return crossing_point


def is_point_on_the_line(l1_x1, l1_y1, l1_x2, l1_y2, x1, x2):
    a1 = get_angle([l1_x1, l1_y1], [l1_x2, l1_y2], [x1, x2])
    a2 = get_angle([l1_x1, l1_y1], [l1_x2, l1_y2], [x1, x2])
    return a1 < 5 and a2 < 5


def plot_trackpoints(trackpoints: np.array, show=True):
    import matplotlib.pyplot as plt
    for point in trackpoints:
        plt.scatter(point[0], point[1], c="blue")
        plt.scatter(point[2], point[3], c="black")
        plt.scatter(point[4], point[5], c="cyan")
    if show:
        plt.show()
    return
plot_trackpoints.__doc__ = """Takes in a numpy array that in the format provided by
get_track_waypoints and displays them as a scatter chart 
with this colour code - blue - center
black inner
cyan outer"""