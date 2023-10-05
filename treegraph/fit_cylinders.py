import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.decomposition import PCA 
from scipy import optimize
from scipy.spatial.transform import Rotation 
from scipy.stats import variation
from tqdm.autonotebook import tqdm
from treegraph.third_party.available_cpu_count import available_cpu_count
from pandarallel import pandarallel

def run(pc, centres, 
        min_pts=10, 
        ransac_iterations=50, 
        sample=100,
        nb_workers=available_cpu_count(),
        verbose=False):

    for c in centres.columns:
        if 'sf' in c: del centres[c]
    
    node_id = centres[centres.n_points > min_pts].sort_values('n_points').node_id.values

    groupby_ = pc.loc[pc.node_id.isin(node_id)].groupby('node_id')
    pandarallel.initialize(progress_bar=verbose, 
                           use_memory_fs=True,
                           nb_workers=min(len(centres), nb_workers))
    
    # cyl = groupby_.parallel_apply(RANSAC_helper, ransac_iterations)
    cyl = groupby_.parallel_apply(RANSAC_helper_2, ransac_iterations, pc, centres)

    cyl = cyl.reset_index()
    cyl.columns=['node_id', 'result']
    cyl.loc[:, 'sf_radius'] = cyl.result.apply(lambda c: c[0])
    cyl.loc[:, 'sf_cx'] =  cyl.result.apply(lambda c: c[1][0])
    cyl.loc[:, 'sf_cy'] =  cyl.result.apply(lambda c: c[1][1])
    cyl.loc[:, 'sf_cz'] =  cyl.result.apply(lambda c: c[1][2])

    centres = pd.merge(centres, 
                       cyl[['node_id', 'sf_radius', 'sf_cx', 'sf_cy', 'sf_cz']], 
                       on='node_id', 
                       how='left')
    
    return centres

def other_cylinder_fit2(xyz, xm=0, ym=0, xr=0, yr=0, r=1):
    
    from scipy.optimize import leastsq
    
    """
    https://stackoverflow.com/a/44164662/1414831
    
    This is a fitting for a vertical cylinder fitting
    Reference:
    http://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XXXIX-B5/169/2012/isprsarchives-XXXIX-B5-169-2012.pdf
    xyz is a matrix contain at least 5 rows, and each row stores x y z of a cylindrical surface
    p is initial values of the parameter;
    p[0] = Xc, x coordinate of the cylinder centre
    P[1] = Yc, y coordinate of the cylinder centre
    P[2] = alpha, rotation angle (radian) about the x-axis
    P[3] = beta, rotation angle (radian) about the y-axis
    P[4] = r, radius of the cylinder
    th, threshold for the convergence of the least squares
    """   
    
    x = xyz.x
    y = xyz.y
    z = xyz.z
    
    p = np.array([xm, ym, xr, yr, r])

    fitfunc = lambda p, x, y, z: (- np.cos(p[3])*(p[0] - x) - z*np.cos(p[2])*np.sin(p[3]) - np.sin(p[2])*np.sin(p[3])*(p[1] - y))**2 + (z*np.sin(p[2]) - np.cos(p[2])*(p[1] - y))**2 #fit function
    errfunc = lambda p, x, y, z: fitfunc(p, x, y, z) - p[4]**2 #error function 

    est_p, success = leastsq(errfunc, p, args=(x, y, z), maxfev=1000)
    
    return est_p

def RANSACcylinderFitting4(xyz_, iterations=50, plot=False):
    
    if plot:
        ax = plt.subplot(111)
    
    bestFit, bestErr = None, np.inf
    xyz_mean = xyz_.mean(axis=0)
    xyz_ -= xyz_mean

    for i in range(iterations):
        
        xyz = xyz_.copy()
        
        # prepare sample 
        sample = xyz.sample(n=20)
        # sample = xyz.sample(n=max(20, int(len(xyz)*.2))) 
        xyz = xyz.loc[~xyz.index.isin(sample.index)]
        
        x, y, a, b, radius = other_cylinder_fit2(sample, 0, 0, 0, 0, 0)
        centre = (x, y)
        if not np.all(np.isclose(centre, 0, atol=radius*1.05)): continue
        
        MX = Rotation.from_euler('xy', [a, b]).inv()
        xyz[['x', 'y', 'z']] = MX.apply(xyz)
        xyz.loc[:, 'error'] = np.linalg.norm(xyz[['x', 'y']] - centre, axis=1) / radius
        idx = xyz.loc[xyz.error.between(.8, 1.2)].index # 40% of radius is prob quite large
        
        # select points which best fit model from original dataset
        alsoInliers = xyz_.loc[idx].copy()
        if len(alsoInliers) < len(xyz_) * .2: continue # skip if no enough points chosen
        
        # refit model using new params
        x, y, a, b, radius = other_cylinder_fit2(alsoInliers, x, y, a, b, radius)
        centre = [x, y]
        if not np.all(np.isclose(centre, 0, atol=radius*1.05)): continue

        MX = Rotation.from_euler('xy', [a, b]).inv()
        alsoInliers[['x', 'y', 'z']] = MX.apply(alsoInliers[['x', 'y', 'z']])
        # calculate error for "best" subset
        alsoInliers.loc[:, 'error'] = np.linalg.norm(alsoInliers[['x', 'y']] - centre, axis=1) / radius      

        if variation(alsoInliers.error) < bestErr:
        
            # for testing uncomment
            c = Circle(centre, radius=radius, facecolor='none', edgecolor='g')

            bestFit = [radius, centre, c, alsoInliers, MX]
            bestErr = variation(alsoInliers.error)

    if bestFit == None: 
        # usually caused by low number of ransac iterations
        return np.nan, xyz[['x', 'y', 'z']].mean(axis=0).values, np.inf, len(xyz_)
    
    radius, centre, c, alsoInliers, MX = bestFit
    centre[0] += xyz_mean.x
    centre[1] += xyz_mean.y
    centre = centre + [xyz_mean.z]
    
    # for testing uncomment
    if plot:
        
        radius, Centre, c, alsoInliers, MX = bestFit

        xyz_[['x', 'y', 'z']] = MX.apply(xyz_)
        xyz_ += xyz_mean
        ax.scatter(xyz_.x, xyz_.y,  s=1, c='grey')

        alsoInliers[['x', 'y', 'z']] += xyz_mean
        cbar = ax.scatter(alsoInliers.x, alsoInliers.y, s=10, c=alsoInliers.error)
        plt.colorbar(cbar)

        ax.scatter(Centre[0], Centre[1], marker='+', s=100, c='r')
        ax.add_patch(c)
        ax.axis('equal')


    return [radius, centre, bestErr, len(xyz_)]

def NotRANSAC(xyz):
    
    try:
        xyz = xyz[['x', 'y', 'z']]
        pca = PCA(n_components=3, svd_solver='auto').fit(xyz)
        xyz[['x', 'y', 'z']] = pca.transform(xyz)
        radius, centre = other_cylinder_fit2(xyz)
        
        if xyz.z.min() - radius < centre[0] < xyz.z.max() + radius or \
           xyz.y.min() - radius < centre[1] < xyz.y.max() + radius: 
            centre = np.hstack([xyz.x.mean(), centre])
        else:
            centre = xyz.mean().values
        
        centre = pca.inverse_transform(centre)
    except:
        radius, centre = np.nan, xyz[['x', 'y', 'z']].mean(axis=0).values
    
    return [radius, centre, np.inf, len(xyz)]


def RANSAC_helper_2(dcluster, ransac_iterations, pc, centres, plot=False):
    # node_id of current cluster
    nid = np.unique(dcluster.node_id)[0]
    # branch_id of current centre node 
    nbranch = centres[centres.node_id == nid].nbranch.values[0]
    # number of centres (segments) of this branch
    nseg = len(centres[centres.nbranch == nbranch])
    # the sequence id of current centre node in its branch
    ncyl = centres[centres.node_id == nid].ncyl.values[0]
    # print(f'node_id = {nid}, nbranch = {nbranch}, nseg = {nseg}, ncyl = {ncyl}')

    # sample points for cyl fitting
    if nseg == 1:
        samples = dcluster
    if ncyl == 0:  # the first segment of this branch
        node_list = centres[(centres.nbranch == nbranch) & (centres.ncyl.isin([0,1]))].node_id.values
        samples = pc[pc.node_id.isin(node_list)]
    if ncyl == (nseg-1):  # the last segment of this branch
        node_list = centres[(centres.nbranch == nbranch) & (centres.ncyl.isin([nseg-2,nseg-1]))].node_id.values
        samples = pc[pc.node_id.isin(node_list)]
    else:
        node_list = centres[(centres.nbranch == nbranch) & (centres.ncyl.isin([ncyl-1,ncyl+1]))].node_id.values
        samples = pc[pc.node_id.isin(node_list)]
    
    # fit cyl to samples using RANSAC
    if len(samples) == 0: # don't think this is required but....
        cylinder = [np.nan, np.array([np.inf, np.inf, np.inf]), np.inf, len(samples)]
    elif len(samples) <= 10:
        cylinder = [np.nan, samples[['x', 'y', 'z']].mean(axis=0).values, np.inf, len(samples)]
    elif len(samples) <= 20:
        cylinder = NotRANSAC(samples)
    else:
        cylinder = RANSACcylinderFitting4(samples[['x', 'y', 'z']], iterations=ransac_iterations, plot=plot)

    return cylinder


def RANSAC_helper(xyz, ransac_iterations, plot=False):
#     try:
        if len(xyz) == 0: # don't think this is required but....
            cylinder = [np.nan, np.array([np.inf, np.inf, np.inf]), np.inf, len(xyz)]
        elif len(xyz) <= 10:
            cylinder = [np.nan, xyz[['x', 'y', 'z']].mean(axis=0).values, np.inf, len(xyz)]
        elif len(xyz) <= 20:
            cylinder = NotRANSAC(xyz)
        else:
            cylinder = RANSACcylinderFitting4(xyz[['x', 'y', 'z']], iterations=ransac_iterations, plot=plot)
#             if cylinder == None: # again not sure if this is necessary...
#                 cylinder = [np.nan, xyz[['x', 'y', 'z']].mean(axis=0)]
                
#     except:
#         cylinder = [np.nan, xyz[['x', 'y', 'z']].mean(axis=0), np.inf, np.inf]

        return cylinder
