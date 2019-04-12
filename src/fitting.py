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
import itertools as it
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from cylinder_fitting import fit


def hough_circle(arr, radius_min=0.02, radius_max=0.4, radius_step=0.01):

    tol = radius_step / 2

    grid_vals = (arr / radius_step).astype(int) * radius_step
    grid_x = np.arange(np.min(grid_vals[:, 0]) - radius_step,
                       np.max(grid_vals[:, 0]) + radius_step, radius_step)
    grid_y = np.arange(np.min(grid_vals[:, 1]) - radius_step,
                       np.max(grid_vals[:, 1]) + radius_step, radius_step)
    grid = list(it.product(grid_x, grid_y))
    grid = np.array([i for i in grid])

    radii = np.arange(radius_min, radius_max, radius_step)

    accumulator = np.zeros([grid.shape[0], len(radii)], dtype=int)

    nbrs = NearestNeighbors(n_jobs=-1).fit(grid)
    distance, indices = nbrs.radius_neighbors(arr, radius=np.max(radii))

    # Looping over the set of values for distance and indices for each grid
    # point.
    for i, (dist, ids) in enumerate(zip(distance, indices)):
        # Looping over the possible radius values.
        for j, r in enumerate(radii):
            # Checking how many points are in the surface (r +- tol) of a
            # sphere and adding the count to current ith and jth indices in the
            # accumulator.
            mask = (dist >= r - tol) & (dist <= r + tol)
            accumulator[ids[mask], j] = accumulator[ids[mask], j] + 1

    max_count = np.max(accumulator, axis=1)
    max_a = np.argmax(accumulator, axis=1)
    opt_rad = radii[max_a]

    nbrs = NearestNeighbors(n_jobs=-1).fit(grid)
    distance, indices = nbrs.radius_neighbors(grid, radius=np.max(radii))
    sph_mask = np.zeros(grid.shape[0], dtype=bool)
    for i, (dist, ids) in enumerate(zip(distance, indices)):
        count_i = max_count[i]
        if count_i > 0:
            r_i = opt_rad[i]
            mask = dist[1:] <= r_i
            if np.sum(mask) > 1:
                if np.max(max_count[ids[1:][mask]]) <= count_i:
                    sph_mask[i] = True

    return grid[sph_mask], opt_rad[sph_mask]


def fit_sphere(arr):

    pca = PCA(n_components=3, svd_solver='full').fit(arr)

    arr_t = pca.transform(arr)
    arr_t = np.vstack((arr_t[:, 2], arr_t[:, 1], arr_t[:, 0])).T

    _, center, rad, error = fit(arr_t)

    center_inv = pca.inverse_transform(center)

    return center_inv, rad, error


