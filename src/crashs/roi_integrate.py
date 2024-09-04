#!/usr/bin/env python3
import numpy as np
from statsmodels.stats.weightstats import DescrStatsW
import pandas as pd
import argparse

from crashs.vtkutil import *
from crashs.util import CrashsDataRoot, Template

def integrate_over_rois(template:Template, args):

    # Read the mesh, vertices and faces
    mesh = load_vtk(args.mesh)
    x, tri = vtk_get_points(mesh), vtk_get_triangles(mesh)

    # Compute triangle areas
    u = x[tri[:,1],:]-x[tri[:,0],:]
    v = x[tri[:,2],:]-x[tri[:,0],:]
    area = np.sqrt(np.sum(np.cross(u,v)**2,1)) / 2

    # Read the plab array to use for integration
    plab = vtk_get_cell_array(mesh, 'plab')

    # Get the triangle area for each label
    tri_area_by_label = plab * area[:,None]

    # Get the triangle area for each vertex
    vtx_area_by_label = np.zeros((x.shape[0], plab.shape[1]))
    for j in range(3):
        vtx_area_by_label[tri[:,j]] += tri_area_by_label / 3

    # Create the output dataframe
    label_ids = template.get_labels_surface_matching()
    df = pd.DataFrame()
    for key in 'subject','session','scan','side':
        if vars(args)[key] is not None:
            df[key] = [ vars(args)[key] for l in label_ids ]
    df['label'] = label_ids

    # For each vertex array, sample it
    for arr_name in args.array:
        arr = vtk_get_point_array(mesh, arr_name)
        mask = ~np.isnan(arr)
        stat = { k: DescrStatsW(arr[mask], vtx_area_by_label[mask,i]) for (i,k) in enumerate(label_ids) }
        df[f'{arr_name}_mean'] = [ stat[k].mean for k in label_ids ]
        df[f'{arr_name}_median'] = [ stat[k].quantile(0.5, False)[0] for k in label_ids ]
        df[f'{arr_name}_q95'] = [ stat[k].quantile(0.95, False)[0] for k in label_ids ]

    # Save the statistics to the output file
    df.to_csv(args.output, index=False)


# The program launcher
class ROILauncher:

    def __init__(self, parser):
        
        # Parse arguments
        parser.add_argument('-C', '--crashs-data', metavar='dir', type=str,
                            help='Path of the CRASHS data folder, if CRASHS_DATA not set')
        parser.add_argument('-t','--template', type=str, required=True, 
                            help='Name of the CRASHS template (folder in $CRASHS_DATA/templates)')
        parser.add_argument('--subject', help='Optional subject ID to include in the output CSV', default=None)
        parser.add_argument('--session', help='Optional session ID to include in the output CSV', default=None)
        parser.add_argument('--scan', help='Optional scan ID to include in the output CSV', default=None)
        parser.add_argument('--side', help='Optional side (left/right) to include in the output CSV', required=True)
        parser.add_argument('-a', '--array', help='Name of the array(s) to integrate', required=True, nargs='+')
        parser.add_argument('-m', '--mesh', help='Mesh in which to perform integration', required=True)
        parser.add_argument('-o', '--output', help='Output CSV file with the statistics', required=True)

        # The function to run
        parser.set_defaults(func = lambda args : self.run(args))


    def run(self, args):

        # Read the CRASHS template
        cdr = CrashsDataRoot(args.crashs_data)
        template = Template(cdr.find_template(args.template))
        integrate_over_rois(template, args)

