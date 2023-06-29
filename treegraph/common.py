import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from scipy import optimize

def node_angle_f(a, b, c):
    
    # normalise distance between coordinate pairs where b is the central coordinate
    ba = a - b
    bc = c - b
  
    # calculate angle between and length of each vector pair 
    angle_pair = lambda ba, bc: np.arccos(np.dot(bc, ba) / (np.linalg.norm(ba) * np.linalg.norm(bc)))

    return angle_pair(bc.T, ba)#[0][0]


def nn(arr, N):
    
    nbrs = NearestNeighbors(n_neighbors=N+1, algorithm='kd_tree').fit(arr)
    distances, indices = nbrs.kneighbors(arr)
    
    return distances[:, 1]


def update_slice_id(centres, branch_hierarchy, node_id, X):
    
    node = centres.loc[centres.node_id == node_id]
    nbranch = node.nbranch.values[0]
    ncyl = node.ncyl.values[0]
    
    # update slices of same branch above ncyls
    centres.loc[(centres.nbranch == nbranch) & (centres.ncyl >= ncyl), 'slice_id'] += X
    
    # update branches above nbranch
    centres.loc[centres.nbranch.isin(branch_hierarchy[nbranch]['above']), 'slice_id'] += X
    
    return centres, branch_hierarchy


def CPC(pc):
    '''
    Input: 
        pc: pd.DataFrame, point clouds of a cluster (with same node_id)
    Output:
        opt: OptimizeResult object, opt.x is the optimal centre coordinates array
    '''
    pc_coor = pc[['x','y','z']]

    # cost function
    def costf(x):
        d = cdist([x], pc_coor)
        mu = d.mean()
        sigma = np.power((d-mu),2).sum() / d.shape[1]
        penalty = d.shape[1]**2 * mu
        # penalty = 1e4
#         print(penalty)
        cost = d.sum() + penalty * sigma
        return cost

    # initial guess of the cluster centre coords
    centroid = pc_coor.median()

    # minimise cost fun to get optimal para est
    opt = optimize.minimize(costf, centroid, method='BFGS')

    return opt


def nn_dist(pc, n_neighbours=10):
    '''
    Calculate distance of the K-neighbours of each point.

    Inputs:
        pc: pd.DataFrame, 
            input point coordinates
        n_neighbours: int, 
                      number of neighbours to use by default for kneighbours queries

    Outputs:
        dists: ndarray of shape (len(pc), n_neighbours)
               distances to the neighbours of each point
        indices: ndarray of shape (len(pc), n_neighbours)
                 indices of the nearest points in the population matrix
    '''
    if len(pc) <= 2:
        return np.nan
    elif len(pc) <= n_neighbours:
        nn = NearestNeighbors(n_neighbors=2).fit(pc[['x','y','z']])
        dists, indices = nn.kneighbors()    
    else:
        nn = NearestNeighbors(n_neighbors=n_neighbours).fit(pc[['x','y','z']])
        dists, indices = nn.kneighbors()
    return dists, indices


def mean_dNN(pc, n_neighbours=10):
    '''
    Calculate the average distance between each point to its K-neighbours, 
    and take the mean of these average distances for each slice of points.
    '''
    if len(pc) <= 2:
        mean_dnn_per_slice = np.nan
    elif len(pc) <= n_neighbours:
        nn = NearestNeighbors(n_neighbors=2).fit(pc[['x','y','z']])
        dists, indices = nn.kneighbors()
        mean_dnn_per_point = np.mean(dists, axis=1)
        mean_dnn_per_slice = np.mean(mean_dnn_per_point)
    else:
        nn = NearestNeighbors(n_neighbors=n_neighbours).fit(pc[['x','y','z']])
        dists, indices = nn.kneighbors()
        mean_dnn_per_point = np.mean(dists, axis=1)
        mean_dnn_per_slice = np.mean(mean_dnn_per_point)
    
    return mean_dnn_per_slice


# def mean_dNN(pc, n_neighbours=10):
#     if len(pc) < 100:
#         mean_dnn_per_slice = np.nan
#     else:
#         nn = NearestNeighbors(n_neighbors=n_neighbours).fit(pc[['x','y','z']])
#         dists, indices = nn.kneighbors()
#         mean_dnn_per_point = np.mean(dists, axis=1)
#         mean_dnn_per_slice = np.mean(mean_dnn_per_point)
#     return mean_dnn_per_slice


# filter out large jump at the end of a branch
def filt_large_jump(centres, bin_dict=None):
    '''
    Input: 
        centres: pd.DataFrame 
                centres attributes of a specific branch
        bin_dict: dict
                segment bin width (value) of each slice (key)
    Output:
        centres_filt: pd.DataFrame
                    centres attributes after filtering connections with large jump                
    '''   
    # print(f'branch {np.unique(centres.nbranch)[0]}')
    if len(centres) < 2:
        return []
    else:
        # calculate the difference of distance from base
        dfb = centres.distance_from_base.values
        dfb_diff = np.diff(dfb)
        centres.loc[centres.index.values[1]:, 'dfb_diff'] = dfb_diff

        # segment bin width of correpsonding slice
        centres.loc[:, 'bin_width'] = centres.slice_id.apply(lambda x: bin_dict[x])

        # ratio of increased distance to slice width
        centres.loc[centres.index.values[1]:, 'ratio'] = centres.dfb_diff / centres.bin_width
        # large jump nodes, excluding the trunk base
        if centres.nbranch.unique()[0] == 0:
            # cut = centres[(centres.ratio >= 1.5) & (centres.ncyl != 1)].ncyl.values
            cut = centres[(centres.ratio >= 5) & (centres.ncyl != 1)].ncyl.values
        else:
            # cut = centres[centres.ratio >= 1.5].ncyl.values
            cut = centres[centres.ratio >= 5].ncyl.values
        if len(cut) > 0:
            cut = cut[0]
            centres.loc[(centres.ncyl >= cut), 'nbranch'] = -1
            centres = centres[centres.nbranch != -1]
        centres.drop(columns=['dfb_diff', 'bin_width', 'ratio'], inplace=True)

    # delete isolated branch whose parent branch has been filtered out 
    for ix, row in centres.iterrows():
        if row.nbranch != 0:
            if len(centres[centres.node_id == row.pnode]) == 0:
                centres.loc[centres.nbranch == row.nbranch, 'nbranch'] = -1
    centres = centres[centres.nbranch != -1]

    return [centres]


## least squares circle fitting
def distance(centre, xp, yp):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    xc, yc = centre
    dist = np.sqrt((xp-xc)**2 + (yp-yc)**2)
    return dist
    
def func(centre, xp, yp):
    """ 
    calculate the algebraic distance between the 2D points 
    and the mean circle centered at c=(xc, yc) 
    """
    Ri = distance(centre, xp, yp)
    return Ri - Ri.mean()

def least_squares_circle(points):
    # Extract x and y coordinates of the points
    if type(points) == pd.DataFrame:
        xp = points['x'].values
        yp = points['y'].values
        cen_est = points.x.mean(), points.y.mean()
    else:
        xp, yp = [], []
        for i in range(len(points)):
            xp.append(points[i][0])
            yp.append(points[i][1])
        cen_est = np.mean(xp), np.mean(yp)
    
    centre, ier = optimize.leastsq(func, cen_est, args=(xp,yp))
    Ri = distance(centre, xp, yp)
    R = Ri.mean()
    residual = np.mean(Ri - R)
    # residual = np.sqrt(np.mean((Ri - R)**2))

    return centre, R, residual


# function to estimate DBH at a given height, default between 1.27-1.33m
def dbh_est(self, h=1.3, verbose=False, plot=False):
    trunk_nids = self.centres[self.centres.nbranch == 0].node_id.values
    zmin = self.pc.z.min()
    zmax = self.pc.z.max()

    zstart = zmin + h - .03
    zstop = zmin + h + .03

    pc_slice = self.pc[(self.pc.z.between(zstart, zstop)) & 
                        (self.pc.node_id.isin(trunk_nids))]

    # if pc_slice contains too few points, increase the slice height
    if len(self.pc)*.01 < 5:
        minpts = self.min_pts
    else:
        minpts = len(self.pc)*.01
    while (len(pc_slice) < min(50, minpts)) & (zmin <= zmax):
        zstop += .01
        pc_slice = self.pc[(self.pc.z.between(zstart, zstop)) & 
                        (self.pc.node_id.isin(trunk_nids))]
        
    centre, radius, residual = least_squares_circle(pc_slice)
    if verbose:
        print(f'measure height = {zstart-zmin:.3f} ~ {zstop-zmin:.3f} m')
        print(f'xc = {centre[0]:.3f} m, yc = {centre[1]:.3f} m')
        print(f'radius = {radius:.3f} m, residual = {residual:.3f} m')


    # DBH est from point clouds
    dbh_clouds = round(2*radius, 3)

    # DBH est from QSM
    sids = pc_slice.slice_id.unique()
    nids = self.centres[self.centres['slice_id'].isin(sids)].node_id.unique()
    cyl_r = self.cyls[self.cyls['p2'].isin(nids)].radius
    dbh_qsm = round(np.nanmean(2*cyl_r), 3)

    if verbose:
        print(f'DBH_from_clouds = {dbh_clouds} m')
        print(f'DBH_from_qsm_cyls = {dbh_qsm:.3f} m')

    if plot:
        # plot extracted trunk slice and the fitted circle
        ax1 = pc_slice.plot.scatter(x='x',y='z')
        ax2 = pc_slice.plot.scatter(x='x',y='y')

        theta_fit = np.linspace(-np.pi, np.pi, 180)
        x_fit = centre[0] + radius * np.cos(theta_fit)
        y_fit = centre[1] + radius * np.sin(theta_fit)
        ax2.scatter(centre[0], centre[1], s=10, c='r')
        ax2.plot(x_fit, y_fit, 'r--', lw=2)
        ax2.axis('equal')

    return dbh_clouds, dbh_qsm


## function to estimate DAH (diameter above-butress height)
def dah_est(self, verbose=False, plot=False):
    trunk_nids = self.centres[self.centres.nbranch == 0].node_id.values
    zmin = self.pc.z.min()
    zmax = self.pc.z.max()
    residual, radius = 1, 1

    while (residual/radius > 0.001) & (zmin <= zmax):
        pc_slice = self.pc[(self.pc.z.between(zmin+1.27, zmin+1.33)) & 
                           (self.pc.node_id.isin(trunk_nids))]
        zmin += .006
        # if len(pc_slice) < 50:
        if len(self.pc)*.01 < 5:
            minpts = self.min_pts
        else:
            minpts = len(self.pc)*.01
        if len(pc_slice) < min(50, minpts):
            continue
        centre, radius, residual = least_squares_circle(pc_slice)

        if verbose:
            print(f'measure height = {zmin+1.27-self.pc.z.min():.3f} ~ {zmin+1.33-self.pc.z.min():.3f} m')
            print(f'xc = {centre[0]:.3f} m, yc = {centre[1]:.3f} m')
            print(f'radius = {radius:.3f} m, residual = {residual:.3f} m')
            print(f'ratio = {residual/radius:.4f}')

    # DAH est from point clouds
    dah_clouds = round(2*radius, 3)
    # DAH est from QSM
    sids = pc_slice.slice_id.unique()
    nids = self.centres[self.centres['slice_id'].isin(sids)].node_id.unique()
    cyl_r = self.cyls[self.cyls['p2'].isin(nids)].radius
    dah_qsm = np.nanmean(2*cyl_r)
    if verbose:
        print(f'DAH_from_clouds = {dah_clouds} m')
        print(f'DAH_from_qsm_cyls = {dah_qsm:.3f} m')
    
    if plot:
        # plot extracted trunk slice and the fitted circle
        ax1 = pc_slice.plot.scatter(x='x',y='z')
        ax2 = pc_slice.plot.scatter(x='x',y='y')
        
        theta_fit = np.linspace(-np.pi, np.pi, 180)
        x_fit = centre[0] + radius * np.cos(theta_fit)
        y_fit = centre[1] + radius * np.sin(theta_fit)
        ax2.scatter(centre[0], centre[1], s=10, c='r')
        ax2.plot(x_fit, y_fit, 'r--', lw=2)
        ax2.axis('equal')

    return dah_clouds, dah_qsm


class treegraph:
    
    def __init__(self, pc, slice_interval=.05, min_pts=10, base_location=None, verbose=False):

        self.pc = pc.copy()        
        self.slice_interval=slice_interval
        self.min_pts = min_pts
        
        if base_location == None:
            self.base_location = self.pc.z.idxmin()
        else:
            self.base_location = base_location
            
        self.verbose = verbose
            
