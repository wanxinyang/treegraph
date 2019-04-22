# Copyright (c) 2019, Matheus Boni Vicari, treestruct
# All rights reserved.
#
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.


__author__ = "Matheus Boni Vicari"
__copyright__ = "Copyright 2019, treestruct"
__credits__ = ["Matheus Boni Vicari"]
__license__ = "GPL3"
__version__ = "0.11"
__maintainer__ = "Matheus Boni Vicari"
__email__ = "matheus.boni.vicari@gmail.com"
__status__ = "Development"


import argparse
from scripts import single_tree, single_branch

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Generates structs from\
 point clouds of individual trees or branches.')    
    parser.add_argument('--script', dest='script', required=True)
    parser.add_argument('--files', dest='input_files', nargs='*',
                        required=True)
    parser.add_argument('--slice_interval', dest='slice_interval', type=float,
                        default=0.05)
    parser.add_argument('--min_pts', dest='min_pts', type=int, default=5)
    parser.add_argument('--dist_threshold', dest='dist_threshold', type=float,
                        default=0.5)
    parser.add_argument('--down_size', dest='down_size', type=float,
                        default=0.1)
    parser.add_argument('--min_cc_dist', dest='min_cc_dist', default='auto')
    parser.add_argument('--max_cc_dist', dest='max_cc_dist', default='auto')
    parser.add_argument('--output_dir', dest='output_dir', default='')
    
    args = parser.parse_args()

    print('Files detected to process:')
    print(args.input_files)
    print('\n')
    
    if args.script == 'single_tree':
        for f in args.input_files:
            print('Processing file: %s' % f)
            print('\n')
            single_tree(f, args.slice_interval, args.min_pts,
                        args.dist_threshold, args.down_size, args.min_cc_dist,
                        args.max_cc_dist, output_dir=args.output_dir)
    elif args.script == 'single_branch':
        for f in args.input_files:
            single_branch(f, args.slice_interval, args.min_pts,
                          args.dist_threshold, args.down_size, args.min_cc_dist,
                        args.max_cc_dist, output_dir=args.output_dir)
        
