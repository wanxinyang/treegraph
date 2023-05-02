import yaml
import treegraph
from treegraph.scripts import tree2qsm
from glob import glob

# run tree2qsm.py on all inputs combination one after the other
inputs_f = glob('*.yml')
for m in range(len(inputs_f)):
    with open (inputs_f[m]) as fr:
        args = yaml.safe_load(fr)
        for key, item in args.items():
            print(f'{key}: {item}')
        
    tree2qsm.run(data_path=args['data_path'], 
                 output_path=args['output_path'],
                 base_idx=args['base_idx'],
                 min_pts=args['min_pts'], 
                 cluster_size=args['cluster_size'], 
                 tip_width=args['tip_width'], 
                 verbose=args['verbose'],
                 base_corr=args['base_corr'],
                 filtering=args['filtering'],
                 txt_file=args['txt_file'],
                 save_graph=args['save_graph'])