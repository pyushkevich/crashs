import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import pymeshlab
import numpy as np

# Read VTK mesh
def load_vtk(filename):
    rd = vtk.vtkPolyDataReader()
    rd.SetFileName(filename)
    rd.Update()
    return rd.GetOutput()

# Read points from the polydata
def vtk_get_points(pd):
    return vtk_to_numpy(pd.GetPoints().GetData())

# Set points in the polydata
def vtk_set_points(pd, x):
    return pd.GetPoints().SetData(numpy_to_vtk(x))

# Read the faces from the polydata
def vtk_get_triangles(pd):
    return vtk_to_numpy(pd.GetPolys().GetData()).reshape(-1,4)[:,1:]

# Map all point arrays to cell arrays
def vtk_all_point_arrays_to_cell_arrays(pd):
    flt = vtk.vtkPointDataToCellData()
    flt.SetInputData(pd)
    flt.Update()
    return flt.GetOutput()

# Get the names of all the point arrays
def vtk_get_point_arrays(pd):
    x = pd.GetPointData()
    return [ x.GetArray(i).GetName() for i in range(x.GetNumberOfArrays()) ]

# Read a point array
def vtk_get_point_array(pd, name):
    a = pd.GetPointData().GetArray(name)
    return vtk_to_numpy(a) if a is not None else None

# Add a cell array to a mesh
def vtk_set_point_array(pd, name, array, array_type=vtk.VTK_FLOAT):
    a = numpy_to_vtk(array, array_type=array_type)
    a.SetName(name)
    pd.GetPointData().AddArray(a)
    return pd

# Get the names of all the cell arrays
def vtk_get_cell_arrays(pd):
    x = pd.GetCellData()
    return [ x.GetArray(i).GetName() for i in range(x.GetNumberOfArrays()) ]

# Read a cell array
def vtk_get_cell_array(pd, name):
    a = pd.GetCellData().GetArray(name)
    return vtk_to_numpy(a) if a is not None else None

# Add a cell array to a mesh
def vtk_set_cell_array(pd, name, array):
    a = numpy_to_vtk(array)
    a.SetName(name)
    pd.GetCellData().AddArray(a)
    return pd

# Set generic field data
def vtk_set_field_data(pd, name, array):
    a = numpy_to_vtk(array)
    a.SetName(name)
    pd.GetFieldData().AddArray(a)
    return pd

# Map a cell array to a point array
def vtk_cell_array_to_point_array(pd, name):
    cell_to_point = vtk.vtkCellDataToPointData()
    cell_to_point.SetInputData(pd)
    cell_to_point.PassCellDataOn()
    cell_to_point.Update()
    vtk_set_point_array(pd, name, vtk_get_point_array(cell_to_point.GetOutput(), name))

# Make a VTK polydata from vertices and triangles
def vtk_make_pd(v, f):
    pd = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(v))
    pd.SetPoints(pts)
    ca = vtk.vtkCellArray()
    ca.SetCells(f.shape[0], numpy_to_vtk(np.insert(f, 0, 3, axis=1).ravel(), array_type=vtk.VTK_ID_TYPE))
    pd.SetPolys(ca)
    return pd

# Clone an existing PD
def vtk_clone_pd(pd):
    pd_clone = vtk.vtkPolyData()
    pd_clone.DeepCopy(pd)
    return pd_clone

# Save VTK polydata
def save_vtk(pd, filename, binary = False):
    wr=vtk.vtkPolyDataWriter()
    wr.SetFileName(filename)
    wr.SetInputData(pd)
    if binary:
        wr.SetFileTypeToBinary()
    wr.Update()

# Sample a cell array arr_in from point data pd at point locations x
def vtk_sample_cell_array_at_vertices(pd, arr_in, x):
    loc = vtk.vtkCellLocator()
    loc.SetDataSet(pd)
    loc.BuildLocator()
    cellId = vtk.reference(0)
    c = [0.0, 0.0, 0.0]
    subId = vtk.reference(0)
    d = vtk.reference(0.0)
    arr_out = np.zeros((x.shape[0], arr_in.shape[1]))
    for j in range(x.shape[0]):
        loc.FindClosestPoint(x[j,:], c, cellId, subId, d)
        arr_out[j,:] = arr_in[cellId, :]
    return arr_out

# Reduction using pymeshlab
def decimate(v, f, target_faces):

    # Create a pymeshlab mesh and add all the arrays to it
    m = pymeshlab.Mesh(vertex_matrix=v, face_matrix=f)

    # Create a mesh set
    ms = pymeshlab.MeshSet()
    ms.add_mesh(m)

    # Perform decimation
    tf = int(target_faces * f.shape[0]) if target_faces < 1.0 else int(target_faces)
    print(f'Decimating mesh, target: {tf} faces')
    if hasattr(pymeshlab.MeshSet, 'meshing_decimation_quadric_edge_collapse') and callable(getattr(pymeshlab.MeshSet, 'meshing_decimation_quadric_edge_collapse')):
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=tf,
                                                    preserveboundary=True,
                                                    preservenormal=True,
                                                    preservetopology=True,
                                                    planarquadric=True)
    else:
        ms.simplification_quadric_edge_collapse_decimation(targetfacenum=tf,
                                                           preserveboundary=True,
                                                           preservenormal=True,
                                                           preservetopology=True,
                                                           planarquadric=True)
    m0 = ms.mesh(0)
    print(f'Decimation complete, {m0.face_matrix().shape[0]} faces')

    # Create a new pd with the vertices and vaces
    return m0.vertex_matrix(), m0.face_matrix()
    
