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
__version__ = "0.1"
__maintainer__ = "Matheus Boni Vicari"
__email__ = "matheus.boni.vicari@gmail.com"
__status__ = "Development"


from sklearn.cluster import KMeans
import numpy as np
import tlseparation as tls
import networkx as nx
from filtering import max_filt


def wood_skeleton(arr, slice_interval, min_cc_dist=0.05, max_cc_dist=0.1,
                  return_dist=False):

    base_id = np.argmin(arr[:, 2])
    G = tls.utility.array_to_graph(arr, base_id, 3, 100, slice_interval,
                                   slice_interval / 2)
    node_ids, distance, path_dict = tls.utility.extract_path_info(G, base_id)
    nodes = arr[node_ids]
    dist = np.array(distance)

    slice_ids = (dist / slice_interval).astype(int)

    n_slices = len(np.unique(slice_ids))
    hh, bins = np.histogram(slice_ids, n_slices)
    norm_log_hh = np.log(hh) / np.sum(np.log(hh))
    hh_factor = 1 - (norm_log_hh - np.min(norm_log_hh))

    for i in range(len((hh_factor[1:]))):
        if hh_factor[i] == np.min(hh_factor):
            hh_factor[i + 1] = np.min(hh_factor)

    hh_coeff = ((max_cc_dist - min_cc_dist) /
                (np.max(hh_factor) - np.min(hh_factor)) *
                (hh_factor - np.min(hh_factor)) + min_cc_dist)

    centers = []
    slice_points = {}
    center_path_dist = {}
    for i, s in enumerate(np.unique(slice_ids)):
        mask = slice_ids == s
        mask_ids = np.where(mask)[0]
        cc = tls.utility.connected_component(nodes[mask], hh_coeff[i])
        for c in np.unique(cc):
            cmask = c == cc
            center_coords = np.mean(nodes[mask][cmask], axis=0)
            centers.append(center_coords)
            slice_points[tuple(center_coords)] = mask_ids[cmask]
            center_path_dist[tuple(center_coords)] = np.mean(dist[mask][cmask],
                                                             axis=0)

    skel = np.array(centers)

    distances = np.zeros(arr.shape[0])
    for c in skel:
        p = slice_points[tuple(c)]
        try:
            dist, nbr = tls.utility.set_nbrs_knn(c.reshape(1, 3), arr[p], 1)
            for i in range(dist.shape[0]):
                distances[p[i]] = dist[i]
        except:
            pass

    for c in skel:
        p = slice_points[tuple(c)]
        pd = center_path_dist[tuple(c)]
        d = distances[p]
        if (d >= np.quantile(distances, 0.98)).any():
            km = KMeans(2).fit(arr[p])
            labels = km.labels_
            centers = km.cluster_centers_
            for i, l in enumerate(np.unique(labels)):
                mask = labels == l
                slice_points[tuple(centers[i])] = p[mask]
                center_path_dist[tuple(centers[i])] = pd

            slice_points.pop(tuple(c))
            center_path_dist.pop(tuple(c))

    if return_dist:
        return slice_points, center_path_dist
    else:
        return slice_points


def upscale_skeleton(skeleton, downsample_cloud, original_cloud):

    nbrs_ids = tls.utility.set_nbrs_knn(downsample_cloud, original_cloud, 1,
                                        False)
    upscale_ids = {}
    for i, n in enumerate(nbrs_ids):
        if n[0] in upscale_ids:
            upscale_ids[n[0]].append(i)
        else:
            upscale_ids[n[0]] = [i]

    new_ref = {}

    for k, v in skeleton.iteritems():
        for vi in v:
            u = upscale_ids[vi]
            for ui in u:
                new_ref.setdefault(k, []).append(ui)

    return new_ref


def smooth_path_spheres(coords, radius, dist, slice_interval):

    slice_ids = (dist / slice_interval).astype(int)

    uids = np.unique(slice_ids)
    pair = {}
    pair_dist = {}
    for u in range(len(uids[:-1])):
        mask_current = np.where(slice_ids == uids[u])[0]
        mask_next = np.where(slice_ids >= uids[u])[0]
        nbr_dist, nbr_ids = tls.utility.set_nbrs_knn(coords[mask_current],
                                                     coords[mask_next], 1,
                                                     return_dist=True)
        nbr_ids = nbr_ids.astype(int)
        for i, (ni, nd) in enumerate(zip(nbr_ids, nbr_dist)):
            if nd > 0:
                current_id = mask_next[i]
                back_id = mask_current[ni][0]
                if current_id in pair:
                    pair[current_id].append(back_id)
                    pair_dist[current_id].append(nd)
                else:
                    pair[current_id] = [back_id]
                    pair_dist[current_id] = [nd]

    new_radius = {}
    for k, v in pair.iteritems():
        min_id = np.argmin(pair_dist[k])
        min_rad = np.mean([radius[k], radius[v[min_id]]])
        new_radius.setdefault(k, []).append(min_rad)
        new_radius.setdefault(v[min_id], []).append(min_rad)

    return new_radius


def skeleton_path_distance(skeleton_coords):

    base_id = np.argmin(skeleton_coords[:, 2])
    G = tls.utility.array_to_graph(skeleton_coords, base_id, 3, 100, 0.15,
                                   0.05)
    node_ids, distance, path_dict = tls.utility.extract_path_info(G, base_id)

    return np.array(distance)


def get_path_nbrs(coords, radius, dist, slice_interval):

    slice_ids = (dist / slice_interval).astype(int)

    uids = np.unique(slice_ids)
    pair = {}
    pair_dist = {}
    for u in range(len(uids[:-1])):
        mask_current = np.where(slice_ids == uids[u])[0]
        mask_next = np.where(slice_ids >= uids[u])[0]
        nbr_dist, nbr_ids = tls.utility.set_nbrs_knn(coords[mask_current],
                                                     coords[mask_next], 1,
                                                     return_dist=True)
        nbr_ids = nbr_ids.astype(int)
        for i, (ni, nd) in enumerate(zip(nbr_ids, nbr_dist)):
            if nd > 0:
                current_id = mask_next[i]
                back_id = mask_current[ni][0]
                if current_id in pair:
                    pair[current_id].append(back_id)
                    pair_dist[current_id].append(nd)
                else:
                    pair[current_id] = [back_id]
                    pair_dist[current_id] = [nd]

    path_nbrs = {}
    for k, v in pair.iteritems():
        min_id = np.argmin(pair_dist[k])
        path_nbrs.setdefault(k, []).append(v[min_id])
        path_nbrs.setdefault(v[min_id], []).append(k)

    return path_nbrs


def skeleton_path(arr, base_id, dist, slice_interval):

    slice_ids = (dist / slice_interval).astype(int)

    G_skeleton = nx.Graph()

    uids = np.unique(slice_ids)
    pair = {}
    pair_dist = {}
    for u in range(len(uids[:-1])):
        mask_current = np.where(slice_ids == u)[0]
        mask_next = np.where(slice_ids >= u)[0]
        if len(mask_current) > 0:
            nbr_dist, nbr_ids = tls.utility.set_nbrs_knn(arr[mask_current],
                                                         arr[mask_next], 1,
                                                         return_dist=True)
            nbr_ids = nbr_ids.astype(int)
            for i, (ni, nd) in enumerate(zip(nbr_ids, nbr_dist)):
                if nd > 0:
                    current_id = mask_next[i]
                    back_id = mask_current[ni][0]
                    if current_id in pair:
                        pair[current_id].append(back_id)
                        pair_dist[current_id].append(nd)
                    else:
                        pair[current_id] = [back_id]
                        pair_dist[current_id] = [nd]

    # WOOD STRUCTURE UP UNTIL HERE
    for k, v in pair.iteritems():
        d = pair_dist[k]
        min_id = np.argmin(d)
        G_skeleton.add_weighted_edges_from([(k, v[min_id], d[min_id])])

    path_distance, path_ids = nx.single_source_bellman_ford(G_skeleton,
                                                            base_id)

    return path_ids, path_distance


def min_radius_path(radius_dict, path_ids):
    new_radius = {}
    for k, v in path_ids.iteritems():
        v = np.array(v)
        mask = np.in1d(v, radius_dict.keys())
        rr_v = np.array([radius_dict[i] for i in v[mask]])
        rr_v2 = max_filt(rr_v)
        for i, vm in enumerate(v[mask]):
            new_radius.setdefault(vm, []).append(rr_v2[i])

    return new_radius
