import argparse
import yaml
from glob import glob

parser = argparse.ArgumentParser()
# required input arguments
parser.add_argument('--data', '-d', type=str, required=True, 
                    help='path to point clouds')
parser.add_argument('--outputs', '-o', type=str, required=True, 
                    help='path to save output files')
# optional input arguments
parser.add_argument('--min_pts', type=int, default=5, required=False, 
                    help='min number of points to pass the filtering')
parser.add_argument('--dbh_height', type=float, default=1.3, required=False, 
                    help='height of DBH estimate, unit in metre, default 1.3m')
parser.add_argument('--cluster_size', type=float, default=.04, required=False, 
                    help='voxel length for downsampling points when generating initial graph')
parser.add_argument('--tip_width', type=float, default=None, required=False, 
                    help='average branch tip diameter (if known), float, unit in metre')
parser.add_argument('--no-base_corr', dest='base_corr', action='store_false', required=False, 
                    help='perform a base fitting correction, default True')
parser.add_argument('--base_idx', type=int, default=None, required=False, 
                    help='index of base point, used if base is not the lowest point, if use this, base_corr should be False')
parser.add_argument('--no-txt_file', dest='txt_file', action='store_false', required=False, 
                    help='produce a text file report, default True')
parser.add_argument('--save_graph', action='store_true', required=False, 
                    help='save the initial distance graph for development purpose, user can keep it False')
parser.add_argument('--verbose', action='store_true', required=False, 
                    help='print something')

args = parser.parse_args()

''' 
Purpose: generate input yaml files for all point clouds in the given path with the given parameters.
'''

for fp in glob(args.data):
    # fp: path to input tree point cloud, str
    print(f'clouds path: {fp}')

    inputs = {'data_path':fp,
              'output_path':args.outputs,
              'base_idx':args.base_idx,
              'min_pts':args.min_pts,
              'cluster_size':args.cluster_size,
              'tip_width':args.tip_width,
              'verbose':args.verbose,
              'base_corr':args.base_corr,
              'dbh_height':args.dbh_height,
              'txt_file':args.txt_file,
              'save_graph':args.save_graph}
    
    treeid = fp.split('/')[-1].split('.')[0]
    ofn = f'{treeid}-inputs-cs{args.cluster_size}-tip{args.tip_width}.yml'

    with open(ofn, 'w') as f:
        f.write(yaml.safe_dump(inputs))
    print(f'generate input file: {ofn}\n')
