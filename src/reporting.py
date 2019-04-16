#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 14:39:28 2019

@author: matheus
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from data_utils import (load_struct, load_ply)


def pdf_report(out_file, cloud_file, struct_file, ply_file):
    
    assert out_file.split('.')[-1] == 'pdf', 'Output file is not a PDF.'

    try:
        import mayavi.mlab as mlab
    except ImportError:
       raise ImportError("Mayavi is required to use this module.")

    # Loading ply file into a triangular mesh (vertices, facets and indices).
    vertex_coords, vertex_ids, facets_tri = load_ply(ply_file)
    f = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(720, 1000))
    mlab.triangular_mesh(vertex_coords[:, 0], vertex_coords[:, 1],
                         vertex_coords[:, 2], facets_tri, scalars=vertex_ids)
    cam = f.scene.camera
    cam.zoom(1.7)
    mlab.draw()
    mlab.savefig('temp_model.png')
    mlab.close()
    img = plt.imread('temp_model.png')

    struct = load_struct(struct_file)
    parameters = struct['input_parameters']
    
    cyl_data = struct['cylinders']
    volume = np.sum([c['volume'] for c in cyl_data.values()])
    length = np.sum([c['length'] for c in cyl_data.values()])
    
    top_iter_height = 0.65

    pp = PdfPages(out_file)
    plt.figure(figsize=(11, 7))
    plt.subplot(121)
    plt.imshow(img)
    plt.box(on=None)
    t = plt.gca()
    t.get_xaxis().set_visible(False)
    t.get_yaxis().set_visible(False)
    plt.subplot(122)
    plt.text(0.1, 0.90, 'cloud: %s' % cloud_file)
    plt.text(0.1, 0.85, 'struct: %s' % struct_file)
    plt.text(0.1, 0.80, '3D model: %s' % ply_file)
    plt.text(0.1, 0.75, 'Total volume: %.2E $m^3$' % volume)
    plt.text(0.1, 0.70, 'Total length: %.2E m' % length)
    for i, (k, v) in enumerate(parameters.iteritems()):
        plt.text(0.1, top_iter_height - (i * 0.05), '%s: %s' % (k, v))
    plt.box(on=None)
    t = plt.gca()
    t.get_xaxis().set_visible(False)
    t.get_yaxis().set_visible(False)
    plt.tight_layout()
    pp.savefig(plt.gcf())
    pp.close()
    plt.close()
    
    os.remove('temp_model.png')
        
    return
