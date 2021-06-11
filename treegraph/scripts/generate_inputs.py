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
# print something
verbose = False
# voxel length for downsample when generating initial graph
cluster_size = [.04, .05]
# min number of points to pass the filtering
minpts = 3
# the base of the exponential function which segments the inital graph
exponent = [2]
# min length of a segmented slice
minbin = [.02, .03, .04, .05, .06, .07]
# max length of a segmented slice
maxbin = [.25, .30, .35, .40, .45, .50]
# path to outputs
output_path = '../results/TreeID/'


## store each combination of inputs into a yaml file
for i in range(len(cluster_size)):
    for j in range(len(exponent)):
        for k in range(len(minbin)):
            for l in range(len(maxbin)):
                inputs = {'data_path':data_path,
                          'base_idx':base_idx,
                          'attribute':attribute,
                          'radius':radius,
                          'verbose':verbose,
                          'cluster_size':cluster_size[i],
                          'minpts':minpts,
                          'exponent':exponent[j],
                          'minbin':minbin[k],
                          'maxbin':maxbin[l],
                          'output_path':output_path}
                with open(f'inputs-cs{cluster_size[i]}-e{exponent[j]}-minb{minbin[k]}-maxb{maxbin[l]}-{radius}.yml', 'w') as f:
                    f.write(yaml.safe_dump(inputs))
