import os 
import argparse
import yaml
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('--data', '-d', type=str, required=True, help='path to point clouds')
parser.add_argument('--outputs', '-o', type=str, required=True, help='path to save output files')
parser.add_argument('--base-idx', type=int, default=None, required=False, 
                    help='index of base point, used if base is not the lowest point')
parser.add_argument('--min-pts', type=int, default=5, required=False, 
                    help='min number of points to pass the filtering')
parser.add_argument('--cluster-size', type=float, nargs='*', default=[.4], required=False, 
                    help='voxel length for downsampling points when generating initial graph')
parser.add_argument('--tip-width', type=float, default=None, required=False, 
                    help='average branch tip diameter (if known), float, unit in metre')
parser.add_argument('--no-base-corr', dest='base_corr', action='store_false', required=False, 
                    help='do not perform a base fitting correction')
parser.add_argument('--no-filtering', dest='filtering', action='store_false', required=False, 
                    help='do not filter out large jump in skeleton node connections')
parser.add_argument('--txt-file', dest='txt_file', action='store_true', required=False, 
                    help='produce a text file report')
parser.add_argument('--save-graph', dest='save_graph', action='store_true', required=False, 
                    help='save the graph separately')
parser.add_argument('--verbose', action='store_true', required=False, 
                    help='print something')

args = parser.parse_args()

''' 
Purpose: generate input yaml files for all point clouds in the given path with the given parameters.
'''

for fp in glob(args.data):

    print(f'clouds path: {fp}')
    
    ## store each combination of inputs into a yaml file
    for cs in args.cluster_size:
        inputs = {'data_path':fp,
                'output_path':args.outputs,
                'base_idx':args.base_idx,
                'min_pts':args.min_pts,
                'cluster_size':cs,
                'tip_width':args.tip_width,
                'verbose':args.verbose,
                'base_corr':args.base_corr,
                'filtering':args.filtering,
                'txt_file':args.txt_file,
                'save_graph':args.save_graph}
        treeid = os.path.split(fp)[1][:-4]
        ofn = f'{treeid}-inputs-cs{cs}-tip{args.tip_width}-filter{args.filtering}.yml'
        with open(ofn, 'w') as f:
            f.write(yaml.safe_dump(inputs))
        print(f'generate input file: {ofn}\n')
