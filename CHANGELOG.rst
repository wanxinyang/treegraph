v0.11
-----
- Corrected shortpath analysis on wood skeleton;
- Added MAE as metric of fitness;
- Included input_parameters in the resulting struct file for easier tracking;
- Added new function to load ply files into a set of vertices coords, vertices indices and triangulation indices;
- Added new function to generate a pdf report to summarize results from a model;
- Included report function to unit_test.py;
- New distance metric to detect optimal min_cc_dist and max_cc_dist for full_tree and small_branch modules;
- Corrected bug that omited the base branch (branch that contains point base_id - lowest in the cloud);


v0.1
----

- Initial version for treestruct, containing the core working code;
