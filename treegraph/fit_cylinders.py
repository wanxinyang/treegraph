import numpy as np
import pandas as pd
from sklearn.decomposition import PCA 
from scipy import optimize

from tqdm.autonotebook import tqdm

from treegraph.third_party import cylinder_fitting


def cylinder_fit(self):

#     if 'sf_radius' in self.centres.columns:
#         del self.centres['sf_radius']

    for c in self.centres.columns:
        if 'sf' in c: del self.centres[c]
    
    node_id = self.centres[self.centres.n_points > self.min_pts].sort_values('n_points').node_id.values
#     self.radius = self.pc.loc[self.pc.node_id.isin(node_id)].groupby('node_id').apply(cylinderFitting)
#     self.centres = pd.merge(self.centres, 
#                             pd.DataFrame(self.radius, columns=['sf_radius']), 
#                             on='node_id', how='left')
    
    groupby_ = self.pc.loc[self.pc.node_id.isin(node_id)].groupby('node_id')
    tqdm.pandas(disable=False if self.verbose else True)
    cyl = groupby_.progress_apply(cylinderFitting)
#     cyl = groupby_.progress_apply(partial_circle)
        
    results = pd.DataFrame(data=[c for c in cyl], columns=['sf_radius', 'centre', 'sf_error'])
    results.loc[:, 'node_id'] = cyl.index
    self.centres = pd.merge(self.centres, 
                            results[['sf_radius', 'sf_error', 'node_id']], 
                            on='node_id', 
                            how='left')
    
    for i, c in enumerate(['cx', 'cy', 'cz']):
            self.centres.\
            set_index('node_id').\
            loc[cyl.index, c] = pd.Series(index=cyl.index, 
                                          data=[row[i] for row in results.centre.values])


def PCA_(xyz):
    pca = PCA(n_components=3, svd_solver='full').fit(xyz)
    return pca, pca.transform(xyz)


class R:
    
    def __init__(self, x, y):
        
        self.x = x
        self.y = y
        self.RADIUS()
        
    def calc_R(self, xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((self.x-xc)**2 + (self.y-yc)**2)

    def f_2(self, c):
        """ calculate the algebraic distance between the data 
        points and the mean circle centered at c=(xc, yc) """
        Ri = self.calc_R(*c)
        return Ri - Ri.mean()
    
    def Df_2b(self, c):

        """ Jacobian of f_2b. The axis corresponding to derivatives 
        must be coherent with the col_deriv option of leastsq"""
        
        xc, yc     = c
        df2b_dc    = np.empty((len(c), self.x.size))

        Ri = self.calc_R(xc, yc)
        df2b_dc[ 0] = (xc - self.x)/Ri                   # dR/dxc
        df2b_dc[ 1] = (yc - self.y)/Ri                   # dR/dyc
        df2b_dc       = df2b_dc - df2b_dc.mean(axis=1)[:, np.newaxis]
        
        return df2b_dc

    def RADIUS(self):

        center_estimate = 0, 0
        center_2, ier = optimize.leastsq(self.f_2, 
                                         center_estimate,
                                         Dfun=self.Df_2b, 
                                         col_deriv=True)

        xc_2, yc_2 = center_2
        Ri_2       = self.calc_R(*center_2)
        R_2        = Ri_2.mean()
        residu_2   = sum((Ri_2 - R_2)**2)

        return R_2, xc_2, yc_2, residu_2

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
        _, centre_inv, rad, error = cylinder_fitting.fit(xyz)

    return rad, centre_inv, error
    

def other_cylinder_fit(xyz):
    
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
    
    pca, xyz = PCA_(xyz)

    p = np.array([np.median(xyz[:, 0]), # x centre
                  np.median(xyz[:, 1]), # y centre
                  1.5, # alpha, rotation angle (radian) about the x-axis
                  0.03, # beta, rotation angle (radian) about the y-axis
                  xyz[:, 2].ptp() / 2
                  ])

    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]

    fitfunc = lambda p, x, y, z: (- np.cos(p[3])*(p[0] - x) - z*np.cos(p[2])*np.sin(p[3]) - np.sin(p[2])*np.sin(p[3])*(p[1] - y))**2 + (z*np.sin(p[2]) - np.cos(p[2])*(p[1] - y))**2 #fit function
    errfunc = lambda p, x, y, z: fitfunc(p, x, y, z) - p[4]**2 #error function 

    est_p, success = leastsq(errfunc, p, args=(x, y, z), maxfev=1000)

    return est_p[4]