import sys
import json
import pandas as pd

if __name__ == '__main__':

    branch = json.load(open(sys.argv[1]))
    name = branch['name']
    tree, internode, node, cyls, centres = [pd.read_json(branch[x]) for x in ['tree', 'internode', 'node', 'cyls', 'centres']]

    print('name:\t\t', name)
    print('date:\t\t', branch['created'])
    print('length:\t\t', '{:.2f} m'.format(tree.loc[0]['length']))
    print('volume:\t\t', '{:.4f} m3'.format(tree.loc[0]['vol']))
    print('area:\t\t', '{:.4f} m2'.format(tree.loc[0]['surface_area']))
    print('nodes:\t\t', '{:.0f}'.format(tree.loc[0]['N_nodes']))
    print('internodes:\t', '{:.0f}'.format(tree.loc[0]['N_furcations']))
    print('tips:\t\t', '{:.0f}'.format(tree.loc[0]['N_terminal_nodes']))
    print('mean tip width:\t', '{:.3f} m'.format(tree.loc[0]['mean_tip_diameter']))
    print('mean distance\nbetween tips:\t', '{:.3f} m'.format(tree.loc[0]['dist_between_tips']))
    try:
        print('path length:\t', '{:.3f}'.format(tree.loc[0]['path_length']))
    except:
        pass
