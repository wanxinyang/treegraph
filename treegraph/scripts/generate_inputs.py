import argparse
import yaml
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('--data', '-d', type=str, required=True, help='path to point clouds')
parser.add_argument('--outputs', '-o', type=str, required=True, help='path to save output files')
args = parser.parse_args()

''' 
Purpose: generate input yaml files for all point clouds in the given path with the given parameters.
'''

for fp in glob(args.data):
    # path to input tree point cloud, str
    data_path = fp
    print(f'clouds path: {fp}')
    # path to save output files
    output_path = args.outputs
    # index of base point, int, default None
    base_idx = None
    # min number of points to pass the filtering
    min_pts = 5
    # voxel length for downsampling points when generating initial graph
    cluster_size = [.04]
    # average branch tip diameter (if known), float, unit in metre
    tip_width = None
    # print something when running the programme
    verbose = False
    # if True then do base fitting correction
    base_corr = True
    # height of DBH estimate, unit in metre, default 1.3m
    dbh_height = 1.3
    # if True then generate a txt file to store inputs, intermediate results and tree-level attributes
    txt_file = True
    # if True then save initial graph & skeleton graph in the output json file
    save_graph = False

    
    ## store each combination of inputs into a yaml file
    for i in range(len(cluster_size)):
        inputs = {'data_path':data_path,
                'output_path':output_path,
                'base_idx':base_idx,
                'min_pts':min_pts,
                'cluster_size':cluster_size[i],
                'tip_width':tip_width,
                'verbose':verbose,
                'base_corr':base_corr,
                'dbh_height':dbh_height,
                'txt_file':txt_file,
                'save_graph':save_graph}
        treeid = data_path.split('/')[-1].split('.')[0]
        ofn = f'{treeid}-inputs-cs{cluster_size[i]}-tip{tip_width}.yml'
        with open(ofn, 'w') as f:
            f.write(yaml.safe_dump(inputs))
        print(f'generate input file: {ofn}\n')
