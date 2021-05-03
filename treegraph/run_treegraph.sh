#!/bin/bash

# include parse_yaml function
. parse_yaml.sh

# read yaml file
eval $(parse_yaml inputs.yml "args_")

# access yaml content and run treegraph
# $verbose = true
if $args_verbose; then
    echo python branch2qsm.py \
-b $args_branch \
-a $args_attribute \
-r $args_radius \
--verbose \
-vl $args_cluster_size \
-mp $args_minpts \
-e $args_exponent \
-minbin $args_minbin \
-maxbin $args_maxbin \
-o $args_output_path

    python branch2qsm.py \
-b $args_branch \
-a $args_attribute \
-r $args_radius \
--verbose \
-vl $args_cluster_size \
-mp $args_minpts \
-e $args_exponent \
-minbin $args_minbin \
-maxbin $args_maxbin \
-o $args_output_path

# $verbose = false    
else
    echo python branch2qsm.py \
-b $args_branch \
-a $args_attribute \
-r $args_radius \
-vl $args_cluster_size \
-mp $args_minpts \
-e $args_exponent \
-minbin $args_minbin \
-maxbin $args_maxbin \
-o $args_output_path
    
    python branch2qsm.py \
-b $args_branch \
-a $args_attribute \
-r $args_radius \
-vl $args_cluster_size \
-mp $args_minpts \
-e $args_exponent \
-minbin $args_minbin \
-maxbin $args_maxbin \
-o $args_output_path

fi
