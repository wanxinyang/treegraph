import numpy as np
import pandas as pd

import treegraph.IO
from treegraph.third_party.ply_io import *

class initialise:
    
    def __init__(self, 
                 data_path='/path/to/pointclouds.ply', 
                 output_path='/path/to/outputs/',
                 base_idx=None,
                 min_pts=10, 
                 downsample=.001,
                 cluster_size=.04,
                 tip_width=None,
                 verbose=False,
                 base_corr=True,
                 filtering=True,
                 txt_file=True,
                 save_graph=False
                 ):
    
        """
        data_path: pandas dataframe or path to point cloud in .ply or .txt format
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
        pc = data_path
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
        self.data_path = data_path
        self.output_path = output_path
        self.min_pts = min_pts
        self.cluster_size = cluster_size
        self.tip_width = tip_width
        self.base_corr = base_corr
        self.filtering = filtering
        self.txt_file = txt_file
        self.save_graph = save_graph
        
        # add unique point id
        self.pc.loc[:, 'pid'] = np.arange(len(self.pc))
        self.base_idx = self.pc.loc[self.pc.z.idxmin() if base_idx == None else base_idx].pid