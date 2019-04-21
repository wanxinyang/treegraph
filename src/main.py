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
from downsampling import downsample_cloud
from fitting import fit_sphere
from skeleton import (wood_skeleton, upscale_skeleton, skeleton_path,
                      min_radius_path)
from geometry import direction_vector
from metrics import (lower_quantile_max_cc, upper_quantile_min_cc)


def full_tree(wood, slice_interval, min_pts, down_size=0.1, min_cc_dist='auto',
              max_cc_dist='auto'):

    wood_down, wood_nbrs = downsample_cloud(wood, down_size,
                                            return_neighbors=True)
    
    if max_cc_dist == 'auto':
        max_cc_dist = lower_quantile_max_cc(wood_down)
    if min_cc_dist == 'auto':
        min_cc_dist = upper_quantile_min_cc(wood_down)
    
    skeleton_downsample, skeleton_path_dist = wood_skeleton(wood_down,
                                                            down_size * 2,
                                                            min_cc_dist,
                                                            max_cc_dist,
                                                            return_dist=True)

    skel_center = {}
    skel_ids = {}
    for i, (k, v) in enumerate(skeleton_downsample.iteritems()):
        skel_center[i] = k
        skel_ids[i] = v

    skel_dist = {}
    for i, (k, v) in enumerate(skeleton_path_dist.iteritems()):
        skel_dist[i] = v

    skeleton_original = upscale_skeleton(skel_ids, wood_down, wood)

    dist = np.array(skel_dist.values())
    skel_coords = np.array(skel_center.values())
    base_id = np.argmin(skel_coords[:, 2])

    path_ids, path_distance = skeleton_path(skel_coords, base_id, dist,
                                            slice_interval)

    skel_fit_radius = {}
    skel_fit_center = {}
    skel_fit_error = {}
    for i, (k, v) in enumerate(skeleton_original.iteritems()):
        if len(v) > min_pts:
            center, rad, error = fit_sphere(wood[v])
            skel_fit_radius[i] = rad
            skel_fit_center[i] = center
            skel_fit_error[i] = error

    new_radius = min_radius_path(skel_fit_radius, path_ids)

    r_min = {}
    n_center = {}
    for k, v in new_radius.iteritems():
        r_min[k] = np.min(v)
        n_center[k] = skel_fit_center[k]
    nr = {}
    for k, r1 in r_min.iteritems():
        k_path = path_ids[k][::-1]
        if len(k_path) > 1:
            for kp in k_path[1:]:
                if kp not in skel_fit_center:
                    nr[kp] = r1
                    n_center[kp] = skel_center[kp]
    for k, r1 in r_min.iteritems():
        nr[k] = r1

    skel_cyl_ids = {}
    cyl_p1 = {}
    cyl_p2 = {}
    cyl_rad = {}
    cyl_len = {}
    cyl_theta = {}
    cyl_phi = {}
    cyl_vol = {}
    cyl_mae = {}
    for k1, r1 in nr.iteritems():
        k_path = path_ids[k1][::-1]
        if len(k_path) > 1:
            k2 = k_path[1]
            r2 = nr[k_path[1]]
            if k1 in skel_cyl_ids:
                skel_cyl_ids[k1] += 1
            else:
                skel_cyl_ids[k1] = 1
            if k2 in skel_cyl_ids:
                skel_cyl_ids[k2] += 1
            else:
                skel_cyl_ids[k2] = 1
            c1 = np.array(n_center[k1])
            c2 = np.array(n_center[k2])
            rad = np.mean([r1, r2])
            length = np.linalg.norm(c1 - c2)
            volume = np.pi * (rad ** 2) * length

            direction = direction_vector(c1, c2)
            theta = np.arccos(np.dot(direction, np.array([0, 0, 1])))
            if np.isnan(theta):
                theta = 0
            phi = np.arctan2(direction[1], direction[0])
            if np.isnan(phi):
                phi = 0

            cyl_p1[k1] = np.array(n_center[k1])
            cyl_p2[k1] = np.array(n_center[k2])
            cyl_rad[k1] = rad
            cyl_len[k1] = length
            cyl_theta[k1] = theta
            cyl_phi[k1] = phi
            cyl_vol[k1] = volume
            
            # Calculating goodness of fit using Mean Absolute Error (cyl_mae).
            cyl_pts = np.vstack((wood[skeleton_original[k1]],
                                 wood[skeleton_original[k2]]))
            dpl = []
            for p in cyl_pts:
                dpl.append(np.linalg.norm(np.cross(c2 - c1, c1 - p)) /
                           np.linalg.norm(c2 - c1))
            cyl_mae[k1] = np.mean(dpl - rad)
        

    hierarchy_dict = {}
    for k, v in cyl_p1.iteritems():
        pids = path_ids[k][::-1]
        for j, p in enumerate(pids):
            if j > 0:
                if skel_cyl_ids[p] > 2:
                    hierarchy_dict.setdefault(pids[j - 1], []).append(k)
                    break

    branch_ids = {}
    for k, v in hierarchy_dict.iteritems():
        for vi in v:
            branch_ids[vi] = k
        branch_ids[k] = k

    branch_connect = {}
    for k, v in hierarchy_dict.iteritems():
        pids = path_ids[k][::-1]
        for p in pids[1:]:
            if p in hierarchy_dict.keys():
                branch_connect.setdefault(k, []).append(p)
                break

    cyl_data = {}
    for k, v in cyl_p1.iteritems():
        if k in branch_ids:
            bid = branch_ids[k]
        else:
            bid = -1
        cyl_data[k] = {'p1': cyl_p1[k], 'p2': cyl_p2[k], 'rad': cyl_rad[k],
                       'length': cyl_len[k], 'theta': cyl_theta[k],
                       'phi': cyl_phi[k], 'volume': cyl_vol[k],
                       'branch_id': bid, 'MAE': cyl_mae[k]}

    branch_data = {}
    for k, v in hierarchy_dict.iteritems():
        bvol = np.sum([cyl_vol[i] for i in v])
        blen = np.sum([cyl_len[i] for i in v])
        if k in branch_connect:
            bcon = branch_connect[k]
        else:
            bcon = -1

        branch_data[k] = {'branch_volume': bvol, 'branch_length': blen,
                          'parent_branch': bcon,
                          'cylinder_ids': v}
        
    input_parameters = {'slice_interval': slice_interval, 'min_pts': min_pts,
                        'down_size': down_size, 'min_cc_dist': min_cc_dist,
                        'max_cc_dist': max_cc_dist}

    struct_data = {'cylinders': cyl_data, 'branches': branch_data,
                   'input_parameters': input_parameters}

    return struct_data


def small_branch(wood, slice_interval=0.01, min_pts=5, min_cc_dist='auto',
                 max_cc_dist='auto'):
    
    if max_cc_dist == 'auto':
        max_cc_dist = lower_quantile_max_cc(wood)
    if min_cc_dist == 'auto':
        min_cc_dist = max_cc_dist
        
    skeleton_original, skeleton_path_dist = wood_skeleton(wood, min_cc_dist * 2,
                                                          min_cc_dist, max_cc_dist,
                                                          return_dist=True)

    skel_center = {}
    skel_ids = {}
    for i, (k, v) in enumerate(skeleton_original.iteritems()):
        skel_center[i] = k
        skel_ids[i] = v

    skel_dist = {}
    for i, (k, v) in enumerate(skeleton_path_dist.iteritems()):
        skel_dist[i] = v

    dist = np.array(skel_dist.values())
    skel_coords = np.array(skel_center.values())
    base_id = np.argmin(skel_coords[:, 2])

    path_ids, path_distance = skeleton_path(skel_coords, base_id, dist,
                                            slice_interval)

    skel_fit_radius = {}
    skel_fit_center = {}
    skel_fit_error = {}
    for i, (k, v) in enumerate(skeleton_original.iteritems()):
        if len(v) > min_pts:
            center, rad, error = fit_sphere(wood[v])
            skel_fit_radius[i] = rad
            skel_fit_center[i] = center
            skel_fit_error[i] = error

    new_radius = min_radius_path(skel_fit_radius, path_ids)

    r_min = {}
    n_center = {}
    for k, v in new_radius.iteritems():
        r_min[k] = np.min(v)
        n_center[k] = skel_fit_center[k]
    nr = {}
    for k, r1 in r_min.iteritems():
        k_path = path_ids[k][::-1]
        if len(k_path) > 1:
            for kp in k_path[1:]:
                if kp not in skel_fit_center:
                    nr[kp] = r1
                    n_center[kp] = skel_center[kp]
    for k, r1 in r_min.iteritems():
        nr[k] = r1

    skel_cyl_ids = {}
    cyl_p1 = {}
    cyl_p2 = {}
    cyl_rad = {}
    cyl_len = {}
    cyl_theta = {}
    cyl_phi = {}
    cyl_vol = {}
    for k1, r1 in nr.iteritems():
        k_path = path_ids[k1][::-1]
        if len(k_path) > 1:
            k2 = k_path[1]
            r2 = nr[k_path[1]]
            if k1 in skel_cyl_ids:
                skel_cyl_ids[k1] += 1
            else:
                skel_cyl_ids[k1] = 1
            if k2 in skel_cyl_ids:
                skel_cyl_ids[k2] += 1
            else:
                skel_cyl_ids[k2] = 1
            c1 = np.array(n_center[k1])
            c2 = np.array(n_center[k2])
            rad = np.mean([r1, r2])
            length = np.linalg.norm(c1 - c2)
            volume = np.pi * (rad ** 2) * length

            direction = direction_vector(c1, c2)
            theta = np.arccos(np.dot(direction, np.array([0, 0, 1])))
            if np.isnan(theta):
                theta = 0
            phi = np.arctan2(direction[1], direction[0])
            if np.isnan(phi):
                phi = 0

            cyl_p1[k1] = np.array(n_center[k1])
            cyl_p2[k1] = np.array(n_center[k2])
            cyl_rad[k1] = rad
            cyl_len[k1] = length
            cyl_theta[k1] = theta
            cyl_phi[k1] = phi
            cyl_vol[k1] = volume

    hierarchy_dict = {}
    for k, v in cyl_p1.iteritems():
        pids = path_ids[k][::-1]
        for j, p in enumerate(pids):
            if j > 0:
                if skel_cyl_ids[p] > 2:
                    hierarchy_dict.setdefault(pids[j - 1], []).append(k)
                    break
                elif p == base_id:
                    hierarchy_dict.setdefault(pids[j - 1], []).append(k)

    branch_ids = {}
    for k, v in hierarchy_dict.iteritems():
        for vi in v:
            branch_ids[vi] = k
        branch_ids[k] = k

    branch_connect = {}
    for k, v in hierarchy_dict.iteritems():
        pids = path_ids[k][::-1]
        for p in pids[1:]:
            if p in hierarchy_dict.keys():
                branch_connect.setdefault(k, []).append(p)
                break

    cyl_data = {}
    for k, v in cyl_p1.iteritems():
        if k in branch_ids:
            bid = branch_ids[k]
        else:
            bid = -1
        cyl_data[k] = {'p1': cyl_p1[k], 'p2': cyl_p2[k], 'rad': cyl_rad[k],
                       'length': cyl_len[k], 'theta': cyl_theta[k],
                       'phi': cyl_phi[k], 'volume': cyl_vol[k],
                       'branch_id': bid}

    branch_data = {}
    for k, v in hierarchy_dict.iteritems():
        bvol = np.sum([cyl_vol[i] for i in v])
        blen = np.sum([cyl_len[i] for i in v])
        if k in branch_connect:
            bcon = branch_connect[k]
        else:
            bcon = -1

        branch_data[k] = {'branch_volume': bvol, 'branch_length': blen,
                          'parent_branch': bcon,
                          'cylinder_ids': v}

    input_parameters = {'slice_interval': slice_interval, 'min_pts': min_pts,
                        'min_cc_dist': min_cc_dist, 'max_cc_dist': max_cc_dist}

    struct_data = {'cylinders': cyl_data, 'branches': branch_data,
                   'input_parameters': input_parameters}

    return struct_data



