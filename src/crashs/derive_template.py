from crashs.vtkutil import *
from crashs.util import CrashsDataRoot, Template

def derive_template(template:Template, side:str, target_geometry:str, output_file:str):
    """
    Derive a new template from the given template geometry.
    
    Parameters:
        template (Template): The CRASHS template to derive from.
        target_geometry (str): The target geometry to derive.
        output_file (str): The output file path for the derived template.
    """
    # Load the template mesh for the specified side
    m_template = load_vtk(template.get_mesh(side))
    
    # Load the target geometry
    m_target = load_vtk(target_geometry)
    
    # Check that the topology of the template and target geometry match
    if np.any(vtk_get_triangles(m_template) != vtk_get_triangles(m_target)):
        raise ValueError("The template and target geometry must have the same topology.")

    # Replace the points in the template mesh with those from the target geometry
    vtk_set_points(m_template, vtk_get_points(m_target))

    # Save the derived mesh to the output file
    save_vtk(m_template, output_file)
    
    
# The program launcher
class DeriveTemplateLauncher:

    def __init__(self, parser):
        
        # Parse arguments
        parser.add_argument('-C', '--crashs-data', metavar='dir', type=str,
                            help='Path of the CRASHS data folder, if CRASHS_DATA not set')
        parser.add_argument('-t','--template', type=str, required=True, 
                            help='Name of the CRASHS template (folder in $CRASHS_DATA/templates)')
        parser.add_argument('--side', '-s', help='Side (left/right) of the template', required=True)
        parser.add_argument('-m', '--mesh', help='Mesh containing the target geometry of the derived template', required=True)
        parser.add_argument('-o', '--output', help='Output mesh for the derived template', required=True)

        # The function to run
        parser.set_defaults(func = lambda args : self.run(args))


    def run(self, args):

        # Read the CRASHS template
        cdr = CrashsDataRoot(args.crashs_data)
        template = Template(cdr.find_template(args.template))
        derive_template(template, args.side, args.mesh, args.output)


