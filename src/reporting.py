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
    cam.zoom(1.2)
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


