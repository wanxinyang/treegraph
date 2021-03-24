import numpy as np
import pandas as pd
from sklearn.decomposition import PCA 
from scipy import optimize
from scipy.spatial.transform import Rotation 

from tqdm.autonotebook import tqdm

from treegraph.third_party import cylinder_fitting
from pandarallel import pandarallel

def run(pc, centres, min_pts=10, ransac_iterations=50, verbose=False):
    
    print('running new version')

#     if 'sf_radius' in self.centres.columns:
#         del self.centres['sf_radius']

    for c in centres.columns:
        if 'sf' in c: del centres[c]
    
    node_id = centres[centres.n_points > min_pts].sort_values('n_points').node_id.values

    groupby_ = pc.loc[pc.node_id.isin(node_id)].groupby('node_id')
    pandarallel.initialize(progress_bar=verbose, verbose=2)
    
    cyl = groupby_.parallel_apply(RANSAC_helper, ransac_iterations)
#     cyl = groupby_.apply(RANSAC_helper)
    cyl.columns=['sf_radius', 'centre']
    cyl.reset_index(inplace=True)
    cyl.loc[:, 'sf_cx'] = cyl.centre.apply(lambda c: c[0])
    cyl.loc[:, 'sf_cy'] = cyl.centre.apply(lambda c: c[1])
    cyl.loc[:, 'sf_cz'] = cyl.centre.apply(lambda c: c[2])
    centres = pd.merge(centres, 
                       cyl[['node_id', 'sf_radius', 'sf_cx', 'sf_cy', 'sf_cz']], 
                       on='node_id', 
                       how='left')
    
#     centres.cx = centres.sf_cx
#     centres.cy = centres.sf_cy
#     centres.cz = centres.sf_cz
    
    return centres

def partial_circle(xyz):
        
    pca, xyz = PCA_(xyz[['x', 'y', 'z']].values)
    results = pd.DataFrame(columns=['R_2', 'xc', 'yc', 'residu'])
    
    for i, y in enumerate([xyz[:, 1], xyz[:, 2]]):
        r = R(xyz[:, 0], y)
        results.loc[i, :] = r.RADIUS()

#     if results.R_2.max() > 3:
#         print(results)
        
#     if results.residu.loc[0] < results.residu.loc[1]: 
        
#         centre = np.array([-results.xc.loc[0], 
#                            -results.yc.loc[0], 
#                            xyz[:, 2].mean()])
#         centre_inv = pca.inverse_transform(centre)
#         return results.R_2.loc[0], centre_inv, results.residu.loc[0] 
    
#     else:        
    centre = np.array([-results.xc.loc[1], 
                       xyz[:, 1].mean(),
                       -results.yc.loc[1]])
    centre_inv = pca.inverse_transform(centre)
    return results.R_2.loc[1], centre_inv, results.residu.loc[1]  

def cylinderFitting(xyz, sample=100):
    
    """
    This method uses the cylinder fitting described in the third party tools
    """
    
    xyz = xyz.sample(n=min([sample, len(xyz)]))
    try:
        pca, xyz = PCA_(xyz[['x', 'y', 'z']].values)
        _, centre, rad, error = cylinder_fitting.fit(xyz)
        centre_inv = pca.inverse_transform(centre)
    except:
        centre_inv, rad, error = xyz.mean(), np.nan, np.nan

    return rad, centre_inv, error
    
def other_cylinder_fit2(xyz):
    
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
    
    xm = np.median(xyz.z)
    ym = np.median(xyz.y)
    
    p = np.array([xm, # x centre
                  ym, # y centre
                  0, # alpha, rotation angle (radian) about the x-axis
                  0, # beta, rotation angle (radian) about the y-axis
                  np.ptp(xyz.z) / 2
                  ])

    x = xyz.z
    y = xyz.y
    z = xyz.x

    fitfunc = lambda p, x, y, z: (- np.cos(p[3])*(p[0] - x) - z*np.cos(p[2])*np.sin(p[3]) - np.sin(p[2])*np.sin(p[3])*(p[1] - y))**2 + (z*np.sin(p[2]) - np.cos(p[2])*(p[1] - y))**2 #fit function
    errfunc = lambda p, x, y, z: fitfunc(p, x, y, z) - p[4]**2 #error function 

    est_p, success = leastsq(errfunc, p, args=(x, y, z), maxfev=1000)

    x, y, a, b, rad = est_p
    centre = np.array([x, y])

    return np.abs(rad), centre

def RANSACcylinderFitting3(xyz, N):
    
    bestFit = None
    bestErr = 99999
    
    try:
        for i in range(N):

            idx = np.random.choice(xyz.index, 
                                   size=min(max(10, int(len(xyz) / 10)), N) * 2,
                                   replace=False)
            sample = xyz.loc[idx][['x', 'y', 'z']].copy()
            pca = PCA(n_components=3, svd_solver='auto').fit(sample)
            sample[['x', 'y', 'z']] = pca.transform(sample)

            maybeInliers = sample.loc[idx[:int(len(idx) / 2)]].copy()
            possibleInliers = sample.loc[~sample.index.isin(maybeInliers.index)].copy()

            radius, centre = other_cylinder_fit2(maybeInliers)
            
            if not sample.z.min() - radius < centre[0] < sample.z.max() + radius or \
               not sample.y.min() - radius < centre[1] < sample.y.max() + radius: continue
            
            error = np.abs(np.linalg.norm(possibleInliers[['z', 'y']] - centre, axis=1)) - radius
            alsoInliers = possibleInliers.loc[error < radius * .1] # 1 cm error is prob quite big?

            # if 10% of potential inliers are actual inliers
            if len(alsoInliers) > len(possibleInliers) * .01:

                allInliers = maybeInliers.append(alsoInliers)
                radius, centre = other_cylinder_fit2(allInliers)
                
                if not sample.z.min() - radius < centre[0] < sample.z.max() + radius or \
                   not sample.y.min() - radius < centre[1] < sample.y.max() + radius: continue
                
                error = np.linalg.norm(allInliers[['z', 'y']] - centre, axis=1) - radius

                if error.mean() < bestErr:

                    centre = np.hstack([sample.x.mean(), centre])
                    centre = pca.inverse_transform(centre)

                    bestFit = [radius, centre]
                    bestErr = error.mean()
    except:
        bestFit = [np.nan, xyz[['x', 'y', 'z']].mean(axis=0).vales]
        
    return bestFit

def NotRANSAC(xyz):
    
    try:
        xyz = xyz[['x', 'y', 'z']]
        pca = PCA(n_components=3, svd_solver='auto').fit(xyz)
        xyz[['x', 'y', 'z']] = pca.transform(xyz)
        radius, centre = other_cylinder_fit2(xyz)
        
#         centre = np.hstack([xyz.x.mean(), centre])
        
        if xyz.z.min() - radius < centre[0] < xyz.z.max() + radius or \
           xyz.y.min() - radius < centre[1] < xyz.y.max() + radius: 
            centre = np.hstack([xyz.x.mean(), centre])
        else:
            centre = xyz.mean().values
        
        centre = pca.inverse_transform(centre)
    except:
        radius, centre = [np.nan, xyz[['x', 'y', 'z']].mean(axis=0).vales]
    
    return [radius, centre]

def RANSAC_helper(xyz, N):
    
    try:
#     if len(xyz) == 0:
#         cylinder = [np.nan, np.array([np.inf, np.inf, np.inf])]
        if len(xyz) < 10:
            cylinder = [np.nan, xyz[['x', 'y', 'z']].mean(axis=0).values]
        elif len(xyz) < 50:
            cylinder = NotRANSAC(xyz)
        else:
            cylinder = RANSACcylinderFitting3(xyz, N)
            if cylinder == None: cylinder = None#other_cylinder_fit(xyz)
                
    except:
        cylinder = [np.nan, xyz[['x', 'y', 'z']].mean(axis=0)]

    return cylinder