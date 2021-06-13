import numpy as np


def run(pc, exponent=2, minbin=.005, maxbin=.02):
    
    """
    qunatises `distance_from_base` dependent on an exponential function

    TODO: allow for any function to be used

    """

    # normalise the distance
    pc.loc[:, 'normalised_distance'] = pc.distance_from_base / pc.distance_from_base.max()
        
    # exponential function to map smaller bin with increased distance from base
    bins, n = np.array([]), 50
    while not pc.distance_from_base.max() <= bins.sum() < pc.distance_from_base.max() * 1.05:
        bins = -np.exp(exponent * np.linspace(0, 1, n)) 
        bins = (bins - bins.min()) / bins.ptp() # normalise to multiply by bin width
        bins = (((maxbin - minbin) * bins) + minbin)
        if bins.sum() < pc.distance_from_base.max():
            n += 1
        else: n -= 1
    
    # merge the first three bin widths to enlarge the range of distance_from_base in 1st slice (the base) 
    bins_w = bins[3:]
    bins_w = np.insert(bins_w, 0, [bins[0]+bins[1]+bins[2]])

    # generate unique id "slice_id" for bins
    pc.loc[:, 'slice_id'] = np.digitize(pc.distance_from_base, bins_w.cumsum())
    
    bins = {i: f for i, f in enumerate(bins_w)}
    
    pc = pc.drop(columns=['normalised_distance'])
    
    return pc, bins
