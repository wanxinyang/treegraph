from networkx.readwrite.pajek import teardown_module
import yaml

## set input candidates
# path to tree point cloud, str
data_path = '../data/TreeID.ply'
# index of base point, int, default None
base_idx = None
# file saving attribute, str, default 'nbranch'
attribute = 'nbranch'
# 'sf_radius': surface fitting radius, 'm_radius': smoothed radius
radius = 'm_radius'
# tip diameter, float, unit in metre
tip_width = None
# print something
verbose = False
# voxel length in downsampling before  generating initial graph
cluster_size = [.04]
# min number of points to pass the filtering
minpts = 3
# the coefficient controlling the steepness of the exponential function which segments the inital graph
exponent = [2]
# minimum bin width of a segmented slice
minbin = [.02, .03, .04, .05, .06, .07]
# maximum bin width of a segmented slice
maxbin = [.25, .30, .35, .40, .45, .50]
# path to outputs
output_path = '../results/TreeID/'
# if True then generate a txt file to store inputs, intermediate results and tree-level attributes
txt_file = True
# if True then do base fitting correction
base_corr = True


## store each combination of inputs into a yaml file
for i in range(len(cluster_size)):
    for j in range(len(exponent)):
        for k in range(len(minbin)):
            for l in range(len(maxbin)):
                inputs = {'data_path':data_path,
                          'base_idx':base_idx,
                          'attribute':attribute,
                          'radius':radius,
                          'tip_width':tip_width,
                          'verbose':verbose,
                          'cluster_size':cluster_size[i],
                          'minpts':minpts,
                          'exponent':exponent[j],
                          'minbin':minbin[k],
                          'maxbin':maxbin[l],
                          'output_path':output_path,
                          'txt_file':txt_file,
                          'base_corr':base_corr}
                with open(f'inputs-cs{cluster_size[i]}-e{exponent[j]}-minb{minbin[k]}-maxb{maxbin[l]}-tip{tip_width}.yml', 'w') as f:
                    f.write(yaml.safe_dump(inputs))
