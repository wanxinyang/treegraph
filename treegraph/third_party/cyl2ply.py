
import math
import numpy as np
import sys
import argparse
import pandas as pd

from tqdm.autonotebook import tqdm

# header needed in ply-file
header = ["ply",
          "format ascii 1.0",
          "comment Author: Cornelis",
          "obj_info Generated using Python",
          "element vertex 50",
          "property float x",
          "property float y",
          "property float z",
          "property float 0",
          "element face 96",
          "property list uchar int vertex_indices",
          "end_header"]

# faces as needed in ply-file face is expressed by the 4 vertice IDs of the face
faces = [[3, 0, 3, 2],
         [3, 0, 4, 3],
         [3, 0, 5, 4],
         [3, 0, 6, 5],
         [3, 0, 7, 6],
         [3, 0, 8, 7],
         [3, 0, 9, 8],
         [3, 0, 10, 9],
         [3, 0, 11, 10], 
         [3, 0, 12, 11],
         [3, 0, 13, 12],
         [3, 0, 14, 13],
         [3, 0, 15, 14],
         [3, 0, 16, 15],
         [3, 0, 17, 16],
         [3, 0, 18, 17],
         [3, 0, 19, 18],
         [3, 0, 20, 19],
         [3, 0, 21, 20],
         [3, 0, 22, 21],
         [3, 0, 23, 22],
         [3, 0, 24, 23],
         [3, 0, 25, 24],
         [3, 0, 2, 25],
         [3, 1, 26, 27],
         [3, 1, 27, 28],
         [3, 1, 28, 29],
         [3, 1, 29, 30],
         [3, 1, 30, 31],
         [3, 1, 31, 32],
         [3, 1, 32, 33],
         [3, 1, 33, 34],
         [3, 1, 34, 35],
         [3, 1, 35, 36],
         [3, 1, 36, 37],
         [3, 1, 37, 38],
         [3, 1, 38, 39],
         [3, 1, 39, 40],
         [3, 1, 40, 41],
         [3, 1, 41, 42],
         [3, 1, 42, 43],
         [3, 1, 43, 44],
         [3, 1, 44, 45],
         [3, 1, 45, 46],
         [3, 1, 46, 47],
         [3, 1, 47, 48],
         [3, 1, 48, 49],
         [3, 1, 49, 26],
         [3, 2, 3, 26],
         [3, 26, 3, 27],
         [3, 3, 4, 27],
         [3, 27, 4, 28],
         [3, 4, 5, 28],
         [3, 28, 5, 29],
         [3, 5, 6, 29],
         [3, 29, 6, 30],
         [3, 6, 7, 30],
         [3, 30, 7, 31],
         [3, 7, 8, 31],
         [3, 31, 8, 32],
         [3, 8, 9, 32],
         [3, 32, 9, 33],
         [3, 9, 10, 33],
         [3, 33, 10, 34],
         [3, 10, 11, 34],
         [3, 34, 11, 35],
         [3, 11, 12, 35],
         [3, 35, 12, 36],
         [3, 12, 13, 36],
         [3, 36, 13, 37],
         [3, 13, 14, 37],
         [3, 37, 14, 38],
         [3, 14, 15, 38],
         [3, 38, 15, 39],
         [3, 15, 16, 39],
         [3, 39, 16, 40],
         [3, 16, 17, 40],
         [3, 40, 17, 41],
         [3, 17, 18, 41],
         [3, 41, 18, 42],
         [3, 18, 19, 42],
         [3, 42, 19, 43],
         [3, 19, 20, 43],
         [3, 43, 20, 44],
         [3, 20, 21, 44],
         [3, 44, 21, 45],
         [3, 21, 22, 45],
         [3, 45, 22, 46],
         [3, 22, 23, 46],
         [3, 46, 23, 47],
         [3, 23, 24, 47],
         [3, 47, 24, 48],
         [3, 24, 25, 48],
         [3, 48, 25, 49],
         [3, 25, 2, 49],
         [3, 49, 2, 26]]

def dot(v1,v2):
    '''returns dot-product of two vectors'''
    return sum(p*q for p,q in zip(v1,v2))

def rotation_matrix(A,angle):
    '''returns the rotation matrix'''
    c = math.cos(angle)
    s = math.sin(angle)
    R = [[A[0]**2+(1-A[0]**2)*c, A[0]*A[1]*(1-c)-A[2]*s, A[0]*A[2]*(1-c)+A[1]*s],
         [A[0]*A[1]*(1-c)+A[2]*s, A[1]**2+(1-A[1]**2)*c, A[1]*A[2]*(1-c)-A[0]*s],
         [A[0]*A[2]*(1-c)-A[1]*s, A[1]*A[2]*(1-c)+A[0]*s, A[2]**2+(1-A[2]**2)*c]]
    return R

def load_cyls(cylfile, args):

    cyls = pd.read_csv(cylfile,
                       sep='\t', 
                       names=['radius', 'length', 'sx', 'sy', 'sz', 'ax', 'ay', 'az', 'parent', 'extension', 
                              'branch', 'BranchOrder', 'PositionInBranch', 'added', 'UnmodRadius'])

    if not args.no_branch:
        branch = pd.read_csv(cylfile.replace('cyl', 'branch'),
                             sep='\t',
                             names=['BOrd', 'BPar', 'BVol', 'BLen', 'BAng', 'BHei', 'BAzi', 'BDia']) 

        branch.set_index(branch.index + 1, inplace=True) # otherwise branches are lablelled from 0
        branch_ids = branch[(branch.BLen >= args.min_length) & (branch.BDia >= args.min_radius * 2)].index

    if args.random:

        values = cyls[args.field].unique()
        MAP = {V:i for i, V in enumerate(np.random.choice(values, size=len(values), replace=False))}
        cyls.loc[:, 'COL'] = cyls[args.field].map(MAP)
        args.field = 'COL'

    if args.verbose: print(cyls.head())
    
    pandas2ply(cyls, args.field, cylfile[:-4] + '.ply')
        
def pandas2ply(cyls, field, out):

    n = len(cyls)
    n_vertices = 50 * n
    n_faces = 96 * n
    
    tempvertices = []
    tempfaces = []
    
    add = 0
    for i, (ix, cyl) in tqdm(enumerate(cyls.iterrows()), total=len(cyls)):

        nvertex = 48                       # number of vertices, do not change!
        rad = cyl.radius                   # cylinder radius
        l = cyl.length                     # cylinder length
        startp = [cyl.sx, cyl.sy, cyl.sz]  # startpoint
        axis = [cyl.ax, cyl.ay, cyl.az]    # axis relative to startpoint

        # first the cylinder is created without rotation
        # starting with center of bottom and top circle
    
        p1 = [0.0, 0.0, 0.0]
        p2 = [0.0, 0.0, l]

        degs = np.deg2rad(np.arange(0, 360, 15))
        ps = [p1,p2]

        # add vertices on the bottom and top circle
        for p0 in [p1, p2]:
            for deg in degs:
                x0 = rad*math.cos(deg)+p0[0]
                y0 = rad*math.sin(deg)+p0[1]
                z0 = p0[2]
            
                ps += [[x0,y0,z0]]

        # the following part is adjusted from script in Matlab that does rotation
        u = [0,0,1]
        raxis = [u[1]*axis[2]-axis[1]*u[2],
                 u[2]*axis[0]-axis[2]*u[0],
                 u[0]*axis[1]-axis[0]*u[1]]

        eucl = (axis[0]**2+axis[1]**2+axis[2]**2)**0.5
        euclr = (raxis[0]**2+raxis[1]**2+raxis[2]**2)**0.5

#         if euclr == 0: euclr = np.nan # not sure why this happens
        for i in range(3):
            raxis[i] /= euclr

        angle = math.acos(dot(u,axis)/eucl)

        M = rotation_matrix(raxis, angle)

        for i in range(len(ps)):
            p = ps[i]
            x = p[0]*M[0][0]+p[1]*M[0][1]+p[2]*M[0][2]
            y = p[0]*M[1][0]+p[1]*M[1][1]+p[2]*M[1][2]
            z = p[0]*M[2][0]+p[1]*M[2][1]+p[2]*M[2][2]

            # add start position
            x += startp[0]
            y += startp[1]
            z += startp[2]
            ps[i] = [x,y,z, cyl[field]]
            #if np.any(np.isnan([x, y, z])): print(cyl)
	
        tempvertices += ps
        for row in faces:
            tempfaces += [[row[0]]+[row[i]+add for i in [1,2,3]]]

        add += 50

    header[4] = "element vertex " + str(n_vertices)
    header[8] = "property float {}".format(field)
    header[9] = "element face " + str(n_faces)

    with open(out, 'w') as theFile:
        for i in header:
            theFile.write(i+'\n')
        for p in tempvertices:
            theFile.write(str(p[0])+' '+str(p[1])+' '+str(p[2])+' '+str(p[3])+'\n')
        for f in tempfaces:
            #print f


            theFile.write(str(f[0])+' '+str(f[1])+' '+str(f[2])+' '+str(f[3])+'\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--cyl', nargs='*', help='list of *cyl.txt files')
    parser.add_argument('-f', '--field', default='branch', help='field with which to colour cylinders by')
    parser.add_argument('-rc', '--random', default=False, action='store_true', help='randomise colours')
    parser.add_argument('-r', '--min_radius', default=0, type=float, help='filter branhces by minimum radius')
    parser.add_argument('-l', '--min_length', default=0, type=float, help='filter branches by minimum length')
    parser.add_argument('--no_branch', action='store_true', help='use if no corresponding branch file is available')
    parser.add_argument('--verbose', action='store_true', help='print some stuff to screen')
    args = parser.parse_args()
    
    for x,line in enumerate(args.cyl):             # loops through treelistfile
        name = line.split()[0]
        load_cyls(name, args)
