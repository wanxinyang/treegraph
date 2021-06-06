import numpy as np
import pandas as pd

import treegraph.IO
from treegraph.third_party.ply_io import *

class initialise:
    
    def __init__(self, pc, 
                 downsample=None,
                 min_pts=10, 
                 exponent=2.0,
                 minbin=.05,
                 maxbin=.5,   
                 cluster_size=.1,
                 base_location=None, 
                 columns=['x', 'y', 'z'],
                 verbose=False,
                 attribute='nbranch',
                 radius='sf_radius',
                 output_path='./'):

        
        """
        pc: pandas dataframe or path to point cloud in .ply or .txt format
            If pandas dataframe, columns ['x', 'y', 'z'] must be present.
        downsample: None or float.
            If value is a float the point cloud will be downsampled before
            running treegraph.
        columns: list default ['x', 'y', 'z']
            If pc is a path to a text file then columns names can also
            be passed.
            
        """
        
        self.verbose = verbose
        if self.verbose: import time

        self.downsample = downsample
        
        # read in data
        if isinstance(pc, pd.DataFrame):
            if np.all([c in pc.columns for c in ['x', 'y', 'z']]):
                self.pc = pc
            else:
                raise Exception('pc columns need to be x, y, z, columns found {}'.format(pc.columns))
        elif isinstance(pc, str) and pc.endswith('.ply'):
            self.pc = read_ply(pc)
        elif isinstance(pc, str) and pc.endswith('.txt'):
            sep = ',' if ',' in open(pc, 'r').readline() else ' '
            self.pc = pd.read_csv(pc, sep=sep)
            if len(columns) != len(self.pc.columns):
                raise Exception('pc read from {} has columns {}, expecting {}'.format(pc, self.pc.columns, columns))
            else:
                self.pc.columns = columns
        else:
            raise Exception('pc is not a pandas dataframe nor a path to point cloud')
                    
#         self.slice_interval=slice_interval
        self.min_pts = min_pts
        self.exponent = exponent
        self.minbin = minbin
        self.maxbin = maxbin
        self.cluster_size=cluster_size
        self.attribute = attribute
        self.radius = radius
        self.output_path = output_path
        
        # add unique point id
        self.pc.loc[:, 'pid'] = np.arange(len(self.pc))
        self.base_location = self.pc.loc[self.pc.z.idxmin() if base_location == None else base_location].pid