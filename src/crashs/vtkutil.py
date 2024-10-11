import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import pymeshlab
import numpy as np
import SimpleITK as sitk
import tempfile
import os

# We need to create some aliases for pymeshlab functions and classes because of the
# changing API and issues with compatibility on older systems
class PyMeshLabInterface:

    @staticmethod
    def percentage(x):
        return pymeshlab.PercentageValue(x) if hasattr(pymeshlab, 'PercentageValue') else pymeshlab.Percentage(x)
    
    @staticmethod
    def meshing_isotropic_explicit_remeshing(ms:pymeshlab.MeshSet, **kwargs):
        if hasattr(pymeshlab.MeshSet, 'meshing_isotropic_explicit_remeshing') and callable(getattr(pymeshlab.MeshSet, 'meshing_isotropic_explicit_remeshing')):
            ms.meshing_isotropic_explicit_remeshing(**kwargs)
        else:
            ms.remeshing_isotropic_explicit_remeshing(**kwargs)

    @staticmethod
    def meshing_decimation_quadric_edge_collapse(ms:pymeshlab.MeshSet, **kwargs):
        if hasattr(pymeshlab.MeshSet, 'meshing_decimation_quadric_edge_collapse') and callable(getattr(pymeshlab.MeshSet, 'meshing_decimation_quadric_edge_collapse')):
            ms.meshing_decimation_quadric_edge_collapse(**kwargs)
        else:
            ms.simplification_quadric_edge_collapse_decimation(**kwargs)

    @staticmethod
    def apply_coord_taubin_smoothing(ms:pymeshlab.MeshSet, **kwargs):
        if hasattr(pymeshlab.MeshSet, 'apply_coord_taubin_smoothing') and callable(getattr(pymeshlab.MeshSet, 'apply_coord_taubin_smoothing')):
            ms.apply_coord_taubin_smoothing(**kwargs)
        else:
            ms.taubin_smooth(**kwargs)
            
    @staticmethod
    def meshing_surface_subdivision_loop(ms:pymeshlab.MeshSet, **kwargs):
        if hasattr(pymeshlab.MeshSet, 'meshing_surface_subdivision_loop') and callable(getattr(pymeshlab.MeshSet, 'meshing_surface_subdivision_loop')):
            ms.meshing_surface_subdivision_loop(**kwargs)
        else:
            ms.subdivision_surfaces_loop(**kwargs)

    @staticmethod
    def add_mesh_to_meshset(ms:pymeshlab.MeshSet, v, f):
        # TODO: there is a bug with pymeshlab (https://github.com/cnr-isti-vclab/PyMeshLab/issues/392) where
        # calling the Mesh constructor results in a segfault. This is an inefficient workaround
        pd = vtk_make_pd(v, f)
        handle, fn = tempfile.mkstemp(suffix='mesh.obj')
        os.close(handle)

        w = vtk.vtkOBJWriter()
        w.SetFileName(fn)
        w.SetInputData(pd)
        w.Update()

        # m = pymeshlab.Mesh(vertex_matrix=v, 
        #                   face_matrix=f)
        ms.load_new_mesh(fn)
        os.remove(fn)

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

# Get generic field data
def vtk_get_field_data(pd, name):
    a = pd.GetFieldData().GetArray(name)
    return vtk_to_numpy(a) if a is not None else None

# Map a cell array to a point array
def vtk_cell_array_to_point_array(pd, name):
    cell_to_point = vtk.vtkCellDataToPointData()
    cell_to_point.SetInputData(pd)
    cell_to_point.PassCellDataOn()
    cell_to_point.Update()
    vtk_set_point_array(pd, name, vtk_get_point_array(cell_to_point.GetOutput(), name))

# Make a VTK polydata from vertices and triangles
def vtk_make_pd(v, f=None):
    pd = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(v))
    pd.SetPoints(pts)
    if f is not None:
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
    

# Map an array to new vertex locations
def vtk_sample_point_array_at_vertices(pd_src, array, x_samples):
    # Use the locator to sample from the halfway mesh
    loc = vtk.vtkCellLocator()
    loc.SetDataSet(pd_src)
    loc.BuildLocator()
    result = np.zeros((x_samples.shape[0], array.shape[1]))    
    cellId = vtk.reference(0)
    c = [0.0, 0.0, 0.0]
    subId = vtk.reference(0)
    d = vtk.reference(0.0)
    pcoord = [0.0, 0.0, 0.0]
    wgt = [0.0, 0.0, 0.0]
    xj = [0.0, 0.0, 0.0]
    for j in range(x_samples.shape[0]):
        loc.FindClosestPoint(x_samples[j,:], c, cellId, subId, d)
        cell = pd_src.GetCell(cellId)
        cell.EvaluatePosition(x_samples[j,:], c, subId, pcoord, d, wgt)
        result[j] = np.sum(np.stack([ array[cell.GetPointId(i),:] * w for i, w in enumerate(wgt) ]), 0)
    return result


# Compute the distance from source mesh to target mesh at each vertex of
# the source mesh and return as an array
def vtk_pointset_to_mesh_distance(pd_src, pd_trg):

    # Use the locator to sample from the halfway mesh
    loc = vtk.vtkCellLocator()
    loc.SetDataSet(pd_trg)
    loc.BuildLocator()
    x = vtk_get_points(pd_src)
    x_to_subj = np.zeros_like(x)
    x_dist = np.zeros(x.shape[0])
    
    cellId = vtk.reference(0)
    c = [0.0, 0.0, 0.0]
    subId = vtk.reference(0)
    d = vtk.reference(0.0)
    pcoord = [0.0, 0.0, 0.0]
    wgt = [0.0, 0.0, 0.0]
    for j in range(x.shape[0]):
        loc.FindClosestPoint(x[j,:], c, cellId, subId, d)
        pd_trg.GetCell(cellId).EvaluatePosition(x[j,:], c, subId, pcoord, d, wgt)
        x_dist[j] = np.sqrt(d.get())

    return x_dist



# Apply a 4x4 affine matrix to a 3D mesh and its arrays
def vtk_apply_sform(pd, sform, 
                    point_coord_arrays=[], point_vector_arrays=[],
                    cell_coord_arrays=[], cell_vector_arrays=[]):
    
    # Apply to the points
    vtk_set_points(pd, vtk_get_points(pd) @ sform[:3,:3].T + sform[:3,3:].T)

    # Apply to the coordinate arrays
    for arr in point_coord_arrays:
        vtk_set_point_array(pd, arr, vtk_get_point_array(pd, arr) @ sform[:3,:3].T + sform[:3,3:].T)
    for arr in cell_coord_arrays:
        vtk_set_cell_array(pd, arr, vtk_get_cell_array(pd, arr) @ sform[:3,:3].T + sform[:3,3:].T)

    # Apply to the vector arrays
    for arr in point_vector_arrays:
        vtk_set_point_array(pd, arr, vtk_get_point_array(pd, arr) @ sform[:3,:3].T)
    for arr in cell_vector_arrays:
        vtk_set_cell_array(pd, arr, vtk_get_cell_array(pd, arr) @ sform[:3,:3].T)


# Reduction using pymeshlab
def decimate(v, f, target_faces):

    # Create a mesh set with the input mesh
    ms = pymeshlab.MeshSet()
    PyMeshLabInterface.add_mesh_to_meshset(ms, v, f)

    # Perform decimation
    tf = int(target_faces * f.shape[0]) if target_faces < 1.0 else int(target_faces)
    print(f'Decimating mesh, target: {tf} faces')
    PyMeshLabInterface.meshing_decimation_quadric_edge_collapse(
        ms, targetfacenum=tf, preserveboundary=True, preservenormal=True,
        preservetopology=True, planarquadric=True)
    m0 = ms.mesh(0)
    print(f'Decimation complete, {m0.face_matrix().shape[0]} faces')

    # Create a new pd with the vertices and vaces
    return m0.vertex_matrix(), m0.face_matrix()
    

# Taubin smoothing using MeshLab
def taubin_smooth(v, f, lam, mu, steps):
    
    # Create a pymeshlab mesh and add all the arrays to it    
    ms = pymeshlab.MeshSet()
    PyMeshLabInterface.add_mesh_to_meshset(ms, v, f)

    # Perform Taubin smoothing
    PyMeshLabInterface.apply_coord_taubin_smoothing(ms, lambda_ = lam, mu = mu, stepsmoothnum = steps)

    # Create a new pd with the vertices and vaces
    m0 = ms.mesh(0)
    return m0.vertex_matrix(), m0.face_matrix()


# Map an array to new vertex locations
def vtk_sample_point_array_at_vertices(pd_src, array, x_samples):
    # Use the locator to sample from the halfway mesh
    loc = vtk.vtkCellLocator()
    loc.SetDataSet(pd_src)
    loc.BuildLocator()
    result = np.zeros((x_samples.shape[0], array.shape[1]))    
    cellId = vtk.reference(0)
    c = [0.0, 0.0, 0.0]
    subId = vtk.reference(0)
    d = vtk.reference(0.0)
    pcoord = [0.0, 0.0, 0.0]
    wgt = [0.0, 0.0, 0.0]
    xj = [0.0, 0.0, 0.0]
    for j in range(x_samples.shape[0]):
        loc.FindClosestPoint(x_samples[j,:], c, cellId, subId, d)
        cell = pd_src.GetCell(cellId)
        cell.EvaluatePosition(x_samples[j,:], c, subId, pcoord, d, wgt)
        result[j] = np.sum(np.stack([ array[cell.GetPointId(i),:] * w for i, w in enumerate(wgt) ]), 0)
    return result


# Given a set of sampling locations on a triangle mesh surface, generate arrays of
# vertex indices and weights that allow data from the source mesh to be sampled at
# the sampling locations. This can be used to interpolate point data, coordinates,
# etc from the source mesh or spatial transformations thereof 
def vtk_get_interpolation_arrays_for_sample(pd_src, x_samples):
    
    # Use the locator to sample from the halfway mesh
    loc = vtk.vtkCellLocator()
    loc.SetDataSet(pd_src)
    loc.BuildLocator()

    # Return data: array of vertex indices and weights
    v_res = np.zeros((x_samples.shape[0], 3), dtype=np.int32)
    w_res = np.zeros((x_samples.shape[0], 3), dtype=np.double)

    cellId = vtk.reference(0)
    c = [0.0, 0.0, 0.0]
    subId = vtk.reference(0)
    d = vtk.reference(0.0)
    pcoord = [0.0, 0.0, 0.0]
    wgt = [0.0, 0.0, 0.0]
    xj = [0.0, 0.0, 0.0]
    for j in range(x_samples.shape[0]):
        loc.FindClosestPoint(x_samples[j,:], c, cellId, subId, d)
        cell = pd_src.GetCell(cellId)
        cell.EvaluatePosition(x_samples[j,:], c, subId, pcoord, d, wgt)
        for i, w in enumerate(wgt):
            v_res[j,i], w_res[j,i] = cell.GetPointId(i), w

    return v_res, w_res


def extract_zero_levelset(img_levelset, edge_len_pct=1.0, to_ras=True):
    pix_raw = sitk.GetArrayFromImage(img_levelset)
    img = vtk.vtkImageData()
    img.GetPointData().SetScalars(numpy_to_vtk(pix_raw.flatten(), array_type=vtk.VTK_FLOAT))
    img.SetDimensions(img_levelset.GetSize()[0], img_levelset.GetSize()[1], img_levelset.GetSize()[2]) 

    cube = vtk.vtkMarchingCubes()
    cube.SetInputData(img)
    cube.SetNumberOfContours(1)
    cube.SetValue(0, 0.0)

    tri1 = vtk.vtkTriangleFilter()
    tri1.SetInputConnection(cube.GetOutputPort())
    tri1.PassLinesOff()
    tri1.PassVertsOff()

    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(tri1.GetOutputPort())
    clean.PointMergingOn()
    clean.SetTolerance(0.0)

    tri2 = vtk.vtkTriangleFilter()
    tri2.SetInputConnection(clean.GetOutputPort())
    tri2.PassLinesOff()
    tri2.PassVertsOff()

    tri2.Update()
    pd_cubes = tri2.GetOutput()

    # Apply remeshing to the template
    ms = pymeshlab.MeshSet()
    ms = pymeshlab.MeshSet()
    PyMeshLabInterface.add_mesh_to_meshset(ms, vtk_get_points(pd_cubes), vtk_get_triangles(pd_cubes))
    PyMeshLabInterface.meshing_isotropic_explicit_remeshing(
        ms, targetlen = PyMeshLabInterface.percentage(edge_len_pct))
    v_remesh, f_remesh = ms.mesh(0).vertex_matrix(), ms.mesh(0).face_matrix()
    return vtk_make_pd(v_remesh, f_remesh)


def isotropic_explicit_remeshing(v, f, **kwargs):
    ms = pymeshlab.MeshSet()
    PyMeshLabInterface.add_mesh_to_meshset(ms, v, f)
    PyMeshLabInterface.meshing_isotropic_explicit_remeshing(ms, **kwargs)
    return ms.mesh(0).vertex_matrix(), ms.mesh(0).face_matrix()

def loop_subdivision(v, f, iterations=1):
    ms = pymeshlab.MeshSet()
    PyMeshLabInterface.add_mesh_to_meshset(ms, v, f)
    PyMeshLabInterface.meshing_surface_subdivision_loop(ms, iterations=iterations)
    return ms.mesh(0).vertex_matrix(), ms.mesh(0).face_matrix()
    