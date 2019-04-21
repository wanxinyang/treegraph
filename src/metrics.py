# Copyright (c) 2019, Matheus Boni Vicari, treestruct
# All rights reserved.
#
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.


__author__ = "Matheus Boni Vicari"
__copyright__ = "Copyright 2019, treestruct"
__credits__ = ["Matheus Boni Vicari"]
__license__ = "GPL3"
__version__ = "0.11"
__maintainer__ = "Matheus Boni Vicari"
__email__ = "matheus.boni.vicari@gmail.com"
__status__ = "Development"


import numpy as np
from knnsearch import set_nbrs_knn


def lower_quantile_max_cc(point_cloud):
    
    mask = point_cloud[:, 2] <= np.quantile(point_cloud[:, 2], 0.1)
    lower_points = point_cloud[mask]
    
    dist, ids = set_nbrs_knn(lower_points, lower_points, 3)
    
    return np.max(dist[:, 1:])


def upper_quantile_min_cc(point_cloud):
    
    mask = point_cloud[:, 2] >= np.quantile(point_cloud[:, 2], 0.95)
    lower_points = point_cloud[mask]
    
    dist, ids = set_nbrs_knn(lower_points, lower_points, 2)
    
    return np.median(dist[:, 1:])
