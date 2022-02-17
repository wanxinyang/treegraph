import yaml
from glob import glob
import treegraph
from treegraph.scripts import tree2qsm

# run tree2qsm.py on all inputs combination one after the other
inputs_f = glob('inputs-cs*.yml')
for m in range(len(inputs_f)):
    with open (inputs_f[m]) as fr:
        args = yaml.safe_load(fr)
        for key, item in args.items():
            print(f'{key}: {item}')
        
    tree2qsm.run(args['data_path'], 
                base_idx=args['base_idx'],
                attribute=args['attribute'], 
                radius=args['radius'], 
                verbose=args['verbose'],
                cluster_size=args['cluster_size'], 
                min_pts=args['minpts'], 
                exponent=args['exponent'], 
                minbin=args['minbin'],
                maxbin=args['maxbin'],
                output=args['output_path'])