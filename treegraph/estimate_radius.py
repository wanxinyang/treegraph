import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from treegraph import plots
from tqdm import tqdm

def run(pc, centres, path_ids, dz1=.3, dz2=1., branch_list=None, plot=False):
    '''
    Radius estimation of individual branches.
    
    Inputs:
        - pc: pd.DataFrame
              point clouds attributes for the whole tree
        - centres: pd.DataFrame
                   skeleton nodes attributes for the whole tree
        - path_ids: dict, 
                    keys: node_id, values: path node_ids from current node to base node
        - zinterval: float,
                     vertical height of each segment slice
        - branch_list: list
                       branch id (nbranch) for estimate radius
        - plot: bool
                if True, plot two subplots for each branch
                left: raw radius estimates with errorbar representing cv  
                right: raw radius overlaped with smoothed radius
        
    Outputs:
        - centres: pd.DataFrame with new columns: 
                   sf_radius - raw radius estimate (unit in metre)
                   cv - coefficient variation for each radius estimate
                   sm_radius - smoothed radius estimate (unit in metre)
    '''
    if branch_list is None:
        branch_list = np.unique(centres.nbranch)
    
    # find stem furcation node
    ncyl = centres[centres.ninternode == 0].ncyl.max()
    stem_fur_node = centres[(centres.nbranch == 0) & (centres.ncyl == ncyl)].node_id.values[0]
    
    # loop over each branch
    for nbranch in tqdm(branch_list):
        ## Step 1: estimate radius with 30cm intervals
        pc_transf, taper = p2a_2(pc, centres, path_ids, zinterval=dz1, branch_list=[nbranch], 
                                 plot=False, interp='nearest')
        if (len(pc_transf) < 1) or (len(taper) < 1):
            continue

        # return estimated radius back to centres attributes
        cen_branch = centres[centres.nbranch == nbranch]
        arr1 = np.array(cen_branch.distance_from_base, dtype='float64')
        if nbranch == 0:
            arr1[0] = arr1[1] / 2.
        arr2 = np.array(taper.dfb, dtype='float64')
        cen_branch['dfb_id'] = np.digitize(arr1, arr2, right=True)
        if cen_branch.dfb_id.max() > taper.index.max():
            cen_branch.loc[cen_branch.dfb_id == cen_branch.dfb_id.max(), 'dfb_id'] = taper.index.max()

        centres.loc[centres.nbranch == nbranch, 'sf_radius'] = cen_branch.dfb_id.apply(lambda x: taper[taper.index == x].p2a_mean.values[0])                   
        centres.loc[centres.nbranch == nbranch, 'cv'] = cen_branch.dfb_id.apply(lambda x: taper[taper.index == x].cv.values[0])
        centres.loc[centres.nbranch == nbranch, 'p2a_std'] = cen_branch.dfb_id.apply(lambda x: taper[taper.index == x].p2a_std.values[0])

        if plot:
            # plot sf_radius with unc vs distance from base in current branch
            fig, axs = plt.subplots(1, 2, figsize=(12,4))
            ax = axs.flatten()
            branch = centres[centres.nbranch==nbranch]
            ax[0].scatter(branch.distance_from_base, branch.sf_radius*1e3, s=5, c='r')
            ax[0].plot(branch.distance_from_base, branch.sf_radius*1e3, label='sf_radius')
            # error bar represents the uncertainty of each radius estimate
            ax[0].errorbar(branch.distance_from_base, branch.sf_radius*1e3, 
                           yerr=branch.cv*1e2, ls='none')
            ax[0].set_xlabel('Distance from base (m)', fontsize=12)
            ax[0].set_ylabel('Estimated radius (mm)', fontsize=12)
            ax[0].set_title(f'Branch {nbranch}', fontsize=12)
            ax[0].legend(loc='upper right')
            
        
        ## Step 2: generate smoother radius estimate
        if len(centres[(centres.nbranch == nbranch) & (~np.isnan(centres.sf_radius))]) < 1:
            continue       
        branch = centres[(centres.nbranch == nbranch) & (~np.isnan(centres.sf_radius))].copy()

        if (nbranch != 0): 
            # use path from stem furcation node to branch tip
            tip = branch.sort_values('ncyl').node_id.values[-1]
            path = path_ids[tip]
            fur_id = path.index(stem_fur_node)
            path = path[fur_id+1:]
            branch = centres[(centres.node_id.isin(path)) & (~np.isnan(centres.sf_radius))].copy()

            # ensure child branch radius doesn't exceed twice that of its parent
            segment = branch[branch.nbranch==nbranch]
            if len(segment.parent_node) > 0: 
                parent_node = centres[centres.nbranch == nbranch].parent_node.values[0]
                if len(centres[centres.node_id == parent_node]) != 0:
                    max_radius = centres[centres.node_id == parent_node]['sf_radius'].values[0]
                    cnids = segment[segment.sf_radius > 2*max_radius].node_id.values
                    branch.loc[branch.node_id.isin(cnids), 'sf_radius'] = 2*max_radius
        
        # estimate radius with 1m intervals
        pc_transf, taper = p2a_2(pc, branch, path_ids, zinterval=dz2, branch_list=[nbranch], 
                                 auto=False, plot=False, interp='nearest')
        if (len(pc_transf) < 2) or (len(taper) < 2):
            continue

        # return smoothed radius back to centres attributes
        p2a_smooth = interp1d(taper.dfb, taper.p2a_mean, kind='linear', bounds_error=False)
        dfb = branch.distance_from_base
        branch.loc[:, 'sm_radius'] = p2a_smooth(dfb)
        branch['sm_radius'] = branch['sm_radius'].astype(float)
        branch.loc[np.isnan(branch.sm_radius), 'sm_radius'] = branch[np.isnan(branch.sm_radius)].sf_radius
        centres.loc[centres.node_id.isin(branch.node_id.values), 'sm_radius'] = branch.sm_radius

        if plot:
            # plot sf_radius overlaped with sm_radius
            attr = 'distance_from_base'
            branch = branch[branch.nbranch == nbranch]
            ax[1].plot(branch[attr], branch['sf_radius']*1e3, label=f'sf_radius')
            ax[1].plot(branch[attr], branch['sm_radius']*1e3, label=f'sm_radius')
            ax[1].scatter(branch[attr], branch['sf_radius']*1e3, s=3, c='r')
            ax[1].scatter(branch[attr], branch['sm_radius']*1e3, s=3, c='r')
#             ax[1].errorbar(branch[attr], branch['sf_radius']*1e3, yerr=branch.cv*1e2, ls='none')
            if attr == 'distance_from_base':
                ax[1].set_xlabel('Distance from tree base (m)', fontsize=12)
#                 xmin, ymin = 0, 0
#                 xmax = branch.distance_from_base.max() * 1.03
#                 ymax = branch.sf_radius.max() * 1.03 * 1e3
#                 ax[1].set_xlim(xmin, xmax)
#                 ax[1].set_ylim(ymin, ymax)
            if attr == 'ncyl':
                ax[1].set_xlabel('Sequence of segment in a branch', fontsize=12)
#             ax[1].set_ylabel('Estimated radius (mm)', fontsize=12)
            ax[1].set_title(f'Branch {nbranch}', fontsize=12)
            ax[1].legend(loc='upper right')
#             ax[1].legend(bbox_to_anchor=(1.2,1))

    return centres


def branch_transform_2(pc, centres, path_ids, auto=True, nbranch=0, 
                       interp='nearest', plot=False, save=False, treeid=None):
    '''
    Apply spline interpolation to transform a curved branch into a straight line.
    
    Inputs:
        pc: pd.DataFrame
            point cloud of the branch
        centres: pd.DataFrame 
                 attributes of skeleton nodes of this branch 
        path_ids: dict 
                  key: node_id, values: path from node to base
        auto: boolean, default True 
              if True then use path from stem furcation node to the branch tip
              if False then use path from the branch base to the tip
        nbranch: integer 
                 branch id, specifying which branch to transform  
        interp: string 
                the type of interpolation to apply
        plot: boolean, default is False
              if True then plot the branch before and after transformation
        save: boolean, default is False 
              if True then save transformed point cloud 
        treeid: string 
                treeid in file name when 'save' if True

    Outputs:
        pc_transf: pd.DataFrame
                   transformed point cloud of this branch
    '''
    if auto:
        # find stem furcation node
        ncyl = centres[centres.ninternode == 0].ncyl.max()
        stem_fur_node = centres[(centres.nbranch == 0) & (centres.ncyl == ncyl)].node_id.values[0]

        samples = centres[centres.nbranch == nbranch].copy()
        # use path from trunk 1st furcation to tip if current branch's segments <= 5
        if (nbranch != 0) & (len(samples) <= 5):
            # segments from stem furcation node to branch tip
            tip = centres[(centres.nbranch == nbranch)].sort_values('ncyl').node_id.values[-1]
            path = path_ids[tip]
            fur_id = path.index(stem_fur_node)
            path = path[fur_id+1:]
            if len(path) < 4:
                return []
            samples = centres[centres.node_id.isin(path)].copy()
    else:
        samples = centres[centres.nbranch == nbranch].copy()
    pc = pc[pc.node_id.isin(samples.node_id.values)].copy()

    if plot:
        # plot a specific branch with furcation node highlighted
        nodes = samples.node_id.values
        plots.plot_slice(pc, samples, slice_id=nodes, attr='node_id', 
                         figtitle=f'Branch {nbranch}, pts coloured in cluster')
    
    ### Sample point
    x = samples.cx
    y = samples.cy
    z = samples.cz
    t = samples.distance_from_base 

    if len(x) < 3:  # branch with pts < 3 
        return [] 
    else:
        ### Apply interpolation for each x, y and z
        xx = interp1d(t, x, kind=interp, bounds_error=False)
        yy = interp1d(t, y, kind=interp, bounds_error=False)
        zz = interp1d(t, z, kind=interp, bounds_error=False)

        if plot:
            ### Visualize the interpolation result
            ## 2D view
            fig1, axs = plt.subplots(1,2,figsize=(10, 8))
            fig1.suptitle(f'Branch {nbranch}, interpolated by {interp}', fontsize=14)
            ax = axs.flatten()
            # front view
            ax[0].scatter(x, z, s=10, c='r') # centre nodes
            tt = np.linspace(t.min(), t.max())
            ax[0].plot(xx(tt), zz(tt), c='b', linewidth=2.5) # spline
            ax[0].scatter(pc.x, pc.z, s=0.5, c='grey', alpha=0.5) # point cloud
            # side view
            ax[1].scatter(y, z, s=10, c='r') # centre nodes
            tt = np.linspace(t.min(), t.max())
            ax[1].plot(yy(tt), zz(tt), c='b', linewidth=2.5) # spline
            ax[1].scatter(pc.y, pc.z, s=0.5, c='grey', alpha=0.5) # point cloud
            
            # ## 3D view
            # fig3 = plt.figure(figsize=(10, 10))
            # ax = fig3.add_subplot(projection='3d')
            # ax.scatter(xs=x, ys=y, zs=z, s=10) # centre nodes
            # tt = np.linspace(t.min(), t.max())
            # ax.plot(xx(tt), yy(tt), zz(tt), c='r', linewidth=2.5) # spline
            # ax.scatter(xs=pc.x, ys=pc.y, zs=pc.z, s=0.01, c='grey', alpha=0.2) # point cloud
            # fig3.suptitle(f'Branch {nbranch}, interpolated by {interp}', fontsize=14)
        
        ### remove curve from pc
        pc.loc[:, 'tx'] = pc.x - xx(pc.distance_from_base)
        pc.loc[:, 'ty'] = pc.y - yy(pc.distance_from_base)
        pc.loc[:, 'tz'] = pc.z - zz(pc.distance_from_base)

        if plot:
            # plot as function of distance from base
            fig2, axs = plt.subplots(2,3,figsize=(12,10))
            axs[0,0].scatter(pc.x, pc.distance_from_base, s=0.1)
            axs[0,0].set_xlabel('pc.x', fontsize=12)
            axs[0,0].set_ylabel('distance_from_base', fontsize=12)

            axs[1,0].scatter(pc.tx, pc.distance_from_base, s=0.1)
            axs[1,0].set_xlabel('pc.tx', fontsize=12)
            axs[1,0].set_ylabel('distance_from_base', fontsize=12)

            axs[0,1].scatter(pc.y, pc.distance_from_base, s=0.1)
            axs[0,1].set_xlabel('pc.y', fontsize=12)

            axs[1,1].scatter(pc.ty, pc.distance_from_base, s=0.1)
            axs[1,1].set_xlabel('pc.ty', fontsize=12)

            axs[0,2].scatter(pc.z, pc.distance_from_base, s=0.1)
            axs[0,2].set_xlabel('pc.z', fontsize=12)

            axs[1,2].scatter(pc.tz, pc.distance_from_base, s=0.1)
            axs[1,2].set_xlabel('pc.tz', fontsize=12)
            
            # set X and Y ticks
            xmin = pc.x.min() / 1.05
            xmax = pc.x.max() * 1.05
            ymin = pc.y.min() / 1.05
            ymax = pc.y.max() * 1.05
            zmin = pc.distance_from_base.min() / 1.05
            zmax = pc.distance_from_base.max() * 1.05
            # keep X-axis and Y-axis in same scale
            if (zmax - zmin) > (xmax - xmin):
                xmin = xmin - (zmax - zmin) / 2.
                xmax = xmax + (zmax - zmin) / 2.
            if (zmax - zmin) > (ymax - ymin):
                ymin = ymin - (zmax - zmin) / 2.
                ymax = ymax + (zmax - zmin) / 2.
            axs[0,0].set_xlim(xmin, xmax)
            axs[0,1].set_xlim(ymin, ymax)
            axs[0,0].set_ylim(zmin, zmax)
            axs[0,1].set_ylim(zmin, zmax)
            
            fig2.tight_layout(h_pad=2)
        
        
        if save:
            # distance_from_base then becomes the new Z coordinate when fitting cylinders
            ofn = f'../results/test_transform/{treeid}.branch{nbranch}.transformed.ply'
            IO.write_ply(ofn, pc[['tx', 'ty', 'distance_from_base']].rename(columns={'tx':'x', 'ty':'y', 'distance_from_base':'z'}))

        pc_transf = pc[~np.isnan(pc.tx)][['node_id', 'tx', 'ty', 'distance_from_base']].rename(columns={'tx':'x', 'ty':'y', 'distance_from_base':'z'})
        
        return pc_transf


def p2a_2(pc, centres, path_ids, zinterval=.3, auto=True,  
          branch_list=None, plot=False, interp='nearest'):

    if branch_list is None:
        branch_list = np.unique(centres.nbranch)
    if plot:
        fig, axs = plt.subplots(1,1,figsize=(10,4))
        ax = [axs]
    # loop over each branch
    for nbranch in branch_list:
        taper = pd.DataFrame(columns=['dfb', 'p2a_mean', 'p2a_std', 'cv'])
        
        ## transform branch from curve to straight
        pc_transf = branch_transform_2(pc, centres, path_ids, nbranch=nbranch, auto=auto,
                                       interp=interp, plot=False, save=False)
        if len(pc_transf) == 0: 
            centres = centres.loc[~(centres.nbranch == nbranch)]
            continue
        # calculate point-to-axis distance of each point
        pc_transf.loc[:, 'p2a'] = np.sqrt(pc_transf.x**2 + pc_transf.y**2)

        ## calculate average point-to-axis distance at every dz interval
        # and use this as estismated radius
        dz = zinterval  # reslice points in interval with dz height (unit in meter)

        zmin = pc_transf.z.min()
        zmax = pc_transf.z.max()
        # the branch cannot be segmented into multiple sections with given interval
        if (zmax - zmin) < dz:
            section = pc_transf
            taper.loc[0, 'dfb'] = zmin
            # taper.loc[0, 'p2a_mean'] = section.p2a.mean()
            # filter out distance outside 95th percentile
            pct95 = np.percentile(section.p2a, 95) 
            section = section[section.p2a < pct95]
            p2a_cv = section.p2a.std() / section.p2a.mean()
            taper.loc[0, 'cv'] = p2a_cv
            taper.loc[0, 'p2a_std'] = section.p2a.std()

            if len(section) < 1:
                taper.loc[0, 'p2a_mean'] = np.nan
            elif p2a_cv < 1:
                taper.loc[0, 'p2a_mean'] = section.p2a.mean() - p2a_cv * section.p2a.std()
            else:
                taper.loc[0, 'p2a_mean'] = np.percentile(section.p2a, 25)
        # the branch can be segmented into multiple sections with given interval
        else:
            for i, z in enumerate(np.arange(zmin, zmax+dz, dz)):
                section = pc_transf[(pc_transf.z >= z) & (pc_transf.z < (z+dz))]
                if len(section) <= 1:
                    continue
                # filter out distance outside 95th percentile
                pct95 = np.nanpercentile(section.p2a, 95)  
                section = section[section.p2a < pct95]
                if len(section) <= 1:
                    continue
                p2a_cv = section.p2a.std() / section.p2a.mean()
                taper.loc[i, 'cv'] = p2a_cv
                taper.loc[i, 'p2a_std'] = section.p2a.std()
                taper.loc[i, 'dfb'] = z
                if p2a_cv < 1 :
                    taper.loc[i, 'p2a_mean'] = section.p2a.mean() - p2a_cv * section.p2a.std()
                else:
                    taper.loc[i, 'p2a_mean'] = np.percentile(section.p2a, 25)
        taper.reset_index(drop=True, inplace=True)
            
        if plot:
            ax[0].scatter(taper.dfb, taper.p2a_mean, s=5, c='r')
            ax[0].plot(taper.dfb, taper.p2a_mean, label=f'branch {nbranch}')
            ax[0].errorbar(taper.dfb, taper.p2a_mean, yerr=taper.p2a_std, 
                            fmt='none', ecolor='grey', capsize=1, alpha=0.5)
            ax[0].set_xlabel('distance from base')
            ax[0].set_ylabel('mean point to axis dist')
            ax[0].set_title(f'Radius estimates from point clouds with interval of {zinterval} m')
            # ax[0].legend(loc='upper right')
            ax[0].legend(bbox_to_anchor=(1.13,1))

    return pc_transf, taper


def p2a_displot(pc, centres, path_ids, interval=0.3, branch_list=None, 
                transform_plot=False, displot=False, xlim=None, ylim=None):
    '''
    Plot distribution of point-to-axis distance (p2a) per section per branch.
    Calculate coefficient of variation (CV) of p2a per section per branch.
    
    Inputs:
        - pc: pd.DataFrame, whole tree pc attributes
        - centres: pd.DataFrame, whole tree centres attributes
        - path_ids: dict, whole tree skeleton node path list
        - interval: float, segment height of each section
        - branch_list: list, branch_id(s) 
        - transform_plot: bool, plot branch transform graphs if True
        - displot: bool, plot p2a distribution if True

    Outputs:
        - figures if plot flag(s) is True
        - cv: dict, cv[nbranch][section.z] =  CV of p2a in this section 
    '''

    if branch_list is not None:
        samples = centres[centres.nbranch.isin(branch_list)]
    else:
        samples = centres
    cv = {}
    p2a_std = {}
    
    for nbranch in samples.nbranch.unique():
        cv[nbranch] = {}
        p2a_std[nbranch] = {}
        ## transform branch from curve to straight
        pc_transf = branch_transform_2(pc, centres, path_ids, nbranch=nbranch, 
                                     interp='slinear', plot=transform_plot, save=False)
        if len(pc_transf) == 0: continue
        # calculate point-to-axis distance of each point
        pc_transf.loc[:, 'p2a'] = np.sqrt(pc_transf.x**2 + pc_transf.y**2)

        ## calculate average point-to-axis distance at every dz interval
        # and use this as estismated radius
        dz = interval  # reslice points in interval with dz height (unit in meter)
        taper = pd.DataFrame(columns=['dfb', 'p2a_mean'])

        zmin = pc_transf.z.min()
        zmax = pc_transf.z.max()

        if (zmax - zmin) < dz:
            section = pc_transf
            taper.loc[0, 'dfb'] = zmin
            # taper.loc[0, 'p2a_mean'] = section.p2a.mean()
            # filter out distance outside 95th percentile
            pct95 = np.percentile(section.p2a, 95) 
            section = section[section.p2a < pct95]
            p2a_cv = section.p2a.std() / section.p2a.mean()
            taper.loc[0, 'cv'] = p2a_cv
            taper.loc[0, 'std'] = section.p2a.std()

            if len(section) < 1:
                taper.loc[0, 'p2a_mean'] = np.nan
            elif p2a_cv < 1:
                taper.loc[0, 'p2a_mean'] = section.p2a.mean() - p2a_cv * section.p2a.std()
            else:
                taper.loc[0, 'p2a_mean'] = np.percentile(section.p2a, 25)
            
            # coefficient of variance
            cv[nbranch][zmin] = p2a_cv
            p2a_std[nbranch][zmin] = section.p2a.std()
        
        else:
            for i, z in enumerate(np.arange(zmin, zmax+dz, dz)):
                section = pc_transf[(pc_transf.z >= z) & (pc_transf.z < (z+dz))]
                if len(section) <= 1:
                    continue
                # filter out distance outside 95th percentile
                pct95 = np.nanpercentile(section.p2a, 95)  
                section = section[section.p2a < pct95]
                if len(section) <= 1:
                    continue
                p2a_cv = section.p2a.std() / section.p2a.mean()
                taper.loc[i, 'cv'] = p2a_cv
                taper.loc[i, 'std'] = section.p2a.std()
                taper.loc[i, 'dfb'] = z
                if p2a_cv < 1 :
                    p2a_mean = section.p2a.mean() - p2a_cv * section.p2a.std()
                    taper.loc[i, 'p2a_mean'] = p2a_mean
                else:
                    p2a_mean = np.percentile(section.p2a, 25)
                    taper.loc[i, 'p2a_mean'] = p2a_mean
                
                # coefficient of variance
                cv[nbranch][z] = p2a_cv
                p2a_std[nbranch][z] = section.p2a.std()
                
                if displot:
                    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
                    ax = axs.flatten()

                    # scatter plot of section
                    ax[0].scatter(section.x, section.y, s=5, c='grey', label='point clouds')
                    ax[0].scatter(0, 0, s=10, c='r', label='axis')
                    ax[0].add_artist(plt.Circle((0, 0), p2a_mean, color='r', alpha=0.2))
                    ax[0].set_xlabel('x (m)', fontsize=12)
                    ax[0].set_ylabel('y (m)', fontsize=12)
                    if xlim is None:
                        ax[0].axis('equal')
                    else:
                        ax[0].set_xlim(xlim[0], xlim[1])
                        ax[0].set_ylim(ylim[0], ylim[1])
                    ax[0].tick_params(axis='both', which='major', labelsize=10)
                    z_low = z - zmin
                    z_upp = z + dz - zmin
                    ax[0].set_title(f'Slice from {z_low:.2f} to {(z_upp):.2f} m', fontsize=14)

                    # distribution of p2a distance
                    from scipy.stats import gaussian_kde
                    ax[1].hist(section.p2a, bins=20, density=True)
                    kde = gaussian_kde(section.p2a)
                    x = np.linspace(section.p2a.min(), section.p2a.max(), 100)
                    ax[1].plot(x, kde(x), color='k')
                    ax[1].axvline(p2a_mean, color='r', label=f'p2a_mean\n= {p2a_mean:.2f} m')
                    # ax[1].axvline(np.percentile(section.p2a, 95), color='g', label='95th percentile')
                    ax[1].set_xlabel('Point-to-axis distance (m)', fontsize=12)
                    ax[1].set_ylabel('Density', fontsize=12)
                    if xlim is not None:
                        ax[1].set_xlim(0, xlim[1])
                    ax[1].tick_params(axis='both', which='major', labelsize=10)
                    ax[1].legend(loc='best', fontsize=12)
                    ax[1].set_title(f'std={section.p2a.std():.2f}, cv={p2a_cv:.2f}', fontsize=14)

                    fig.tight_layout()
        
        taper.reset_index(drop=True, inplace=True)      

    return p2a_std, cv


def rad_cv_plot(self, branch_list=[*range(100)], bin_width=0.5, 
                title=None):
    """
    Calculate the mean of cv at each distance from base with a given interval.
    Plot the mean of cv with 95% CI along distance from base.
    """

    p2a_std, cv = p2a_displot(self.pc, self.centres, self.path_ids, interval=.3, 
                              branch_list=branch_list, transform_plot=False, displot=False)
    
    ## calculate the mean of cv at given interval
    # transfer cv dict to dataframe
    data = []
    for branch_id, values in cv.items():
        for dfb, cv_ in values.items():
            data.append((branch_id, dfb, cv_))

    cv_df = pd.DataFrame(data, columns=['branch_id', 'dfb', 'cv'])
                                            
    # segment distance from base into bins anc calculate the mean of cv at each bin
    bins = np.arange(0, cv_df.dfb.max() + bin_width, bin_width)
    cv_df['dfb_bin'] = pd.cut(cv_df.dfb, bins=bins)

    groupdf = cv_df.groupby(['dfb_bin']).agg({'cv': ['mean', 'std']}).reset_index()
    groupdf.columns = ['dfb_bin', 'cv_mean', 'cv_std']
    # convert the bin column to numeric
    groupdf['dfb_bin'] = groupdf.dfb_bin.apply(lambda x: x.mid)
    # the upper and lower bound of the 95% CI for the mean cv
    groupdf['upp'] = groupdf.cv_mean + 1.96 * groupdf.cv_std
    groupdf['low'] = groupdf.cv_mean - 1.96 * groupdf.cv_std

    # plot
    fig = plt.figure(figsize=(8,5))
    # plot the mean cv
    plt.plot(groupdf.dfb_bin, groupdf.cv_mean, label='mean of CV')
    # plot 95% CI threshold
    plt.fill_between(groupdf.dfb_bin, groupdf.low, groupdf.upp, alpha=.3,
                    label='95% CI')
    plt.xlabel('Distance from base (m)', fontsize=12)
    plt.ylabel('CV of radius estimation', fontsize=12)

    # plot main furcation node location
    stem_fur = self.centres[self.centres.ninternode == 0].distance_from_base.max()
    plt.axvline(x=stem_fur, color='green', linestyle='--', alpha=0.3, 
                label='main branching node')
    plt.legend(loc='upper left')
    if title:
        plt.title(title, fontsize=14)

    return fig


def rad_std_plot(self, branch_list=[*range(100)], bin_width=0.5, 
                title=None):
    """
    Calculate the mean of cv at each distance from base with a given interval.
    Plot the mean of cv with 95% CI along distance from base.
    """

    p2a_std, cv = p2a_displot(self.pc, self.centres, self.path_ids, interval=.3, 
                              branch_list=branch_list, transform_plot=False, displot=False)
    
    ## calculate the mean of cv at given interval
    # transfer cv dict to dataframe
    data = []
    for branch_id, values in p2a_std.items():
        for dfb, std_ in values.items():
            data.append((branch_id, dfb, std_))

    std_df = pd.DataFrame(data, columns=['branch_id', 'dfb', 'p2a_std'])
                                            
    # segment distance from base into bins anc calculate the mean of p2a_std at each bin
    bins = np.arange(0, std_df.dfb.max() + bin_width, bin_width)
    std_df['dfb_bin'] = pd.cut(std_df.dfb, bins=bins)

    groupdf = std_df.groupby(['dfb_bin']).agg({'p2a_std': ['mean', 'std']}).reset_index()
    groupdf.columns = ['dfb_bin', 'p2a_std_mean', 'p2a_std_std']
    # convert the bin column to numeric
    groupdf['dfb_bin'] = groupdf.dfb_bin.apply(lambda x: x.mid)
    # the upper and lower bound of the 95% CI for the mean p2a_std
    groupdf['upp'] = groupdf.p2a_std_mean + 1.96 * groupdf.p2a_std_std
    groupdf['low'] = groupdf.p2a_std_mean - 1.96 * groupdf.p2a_std_std

    # plot
    fig = plt.figure(figsize=(8,5))
    # plot the mean cv
    plt.plot(groupdf.dfb_bin, groupdf.p2a_std_mean, label='average SD')
    # plot 95% CI threshold
    plt.fill_between(groupdf.dfb_bin, groupdf.low, groupdf.upp, alpha=.3,
                    label='95% CI')
    plt.xlabel('Distance from base (m)', fontsize=12)
    plt.ylabel('SD of radius estimation', fontsize=12)

    # plot main furcation node location
    stem_fur = self.centres[self.centres.ninternode == 0].distance_from_base.max()
    plt.axvline(x=stem_fur, color='green', linestyle='--', alpha=0.3, 
                label='main branching node')
    plt.legend(loc='upper left')
    if title:
        plt.title(title, fontsize=14)

    return fig
