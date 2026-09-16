# pyright: reportAttributeAccessIssue=false
# pymeshlab ships as a compiled extension with no type stubs/py.typed marker,
# so Pylance/Pyright cannot resolve any of its attributes. All direct use of
# the pymeshlab API in crashs is confined to this file (via PyMeshLabInterface)
# so that this suppression does not need to be repeated elsewhere.
import pymeshlab
import vtk
from vtk.util.numpy_support import numpy_to_vtk
import numpy as np
import tempfile
import os


# We need to create some aliases for pymeshlab functions and classes because of the
# changing API and issues with compatibility on older systems. This class is also
# the single point of contact with the pymeshlab package for the rest of crashs.
class PyMeshLabInterface:

    @staticmethod
    def percentage(x):
        return pymeshlab.PercentageValue(x) if hasattr(pymeshlab, 'PercentageValue') else pymeshlab.Percentage(x)

    @staticmethod
    def absolute_value(x):
        return pymeshlab.AbsoluteValue(x)

    @staticmethod
    def create_meshset() -> pymeshlab.MeshSet:
        return pymeshlab.MeshSet()

    @staticmethod
    def meshing_isotropic_explicit_remeshing(ms: pymeshlab.MeshSet, **kwargs):
        if hasattr(pymeshlab.MeshSet, 'meshing_isotropic_explicit_remeshing') and callable(getattr(pymeshlab.MeshSet, 'meshing_isotropic_explicit_remeshing')):
            ms.meshing_isotropic_explicit_remeshing(**kwargs)
        else:
            ms.remeshing_isotropic_explicit_remeshing(**kwargs)

    @staticmethod
    def meshing_decimation_quadric_edge_collapse(ms: pymeshlab.MeshSet, **kwargs):
        if hasattr(pymeshlab.MeshSet, 'meshing_decimation_quadric_edge_collapse') and callable(getattr(pymeshlab.MeshSet, 'meshing_decimation_quadric_edge_collapse')):
            ms.meshing_decimation_quadric_edge_collapse(**kwargs)
        else:
            ms.simplification_quadric_edge_collapse_decimation(**kwargs)

    @staticmethod
    def apply_coord_taubin_smoothing(ms: pymeshlab.MeshSet, **kwargs):
        if hasattr(pymeshlab.MeshSet, 'apply_coord_taubin_smoothing') and callable(getattr(pymeshlab.MeshSet, 'apply_coord_taubin_smoothing')):
            ms.apply_coord_taubin_smoothing(**kwargs)
        else:
            ms.taubin_smooth(**kwargs)

    @staticmethod
    def meshing_surface_subdivision_loop(ms: pymeshlab.MeshSet, **kwargs):
        if hasattr(pymeshlab.MeshSet, 'meshing_surface_subdivision_loop') and callable(getattr(pymeshlab.MeshSet, 'meshing_surface_subdivision_loop')):
            ms.meshing_surface_subdivision_loop(**kwargs)
        else:
            ms.subdivision_surfaces_loop(**kwargs)

    @staticmethod
    def compute_curvature_principal_directions_per_vertex(ms: pymeshlab.MeshSet, **kwargs):
        ms.compute_curvature_principal_directions_per_vertex(**kwargs)

    @staticmethod
    def create_sphere(ms: pymeshlab.MeshSet, subdiv: int):
        ms.create_sphere(subdiv=subdiv)

    @staticmethod
    def add_mesh_to_meshset(ms: pymeshlab.MeshSet, v, f):
        # TODO: there is a bug with pymeshlab (https://github.com/cnr-isti-vclab/PyMeshLab/issues/392) where
        # calling the Mesh constructor results in a segfault. This is an inefficient workaround
        pd = vtk.vtkPolyData()
        pts = vtk.vtkPoints()
        pts.SetData(numpy_to_vtk(v))
        pd.SetPoints(pts)
        ca = vtk.vtkCellArray()
        ca.SetCells(f.shape[0], numpy_to_vtk(np.insert(f, 0, 3, axis=1).ravel(), array_type=vtk.VTK_ID_TYPE))
        pd.SetPolys(ca)

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

    @staticmethod
    def create_meshset_from_arrays(v, f) -> pymeshlab.MeshSet:
        ms = PyMeshLabInterface.create_meshset()
        PyMeshLabInterface.add_mesh_to_meshset(ms, v, f)
        return ms

    @staticmethod
    def create_meshset_from_arrays_direct(v, f) -> pymeshlab.MeshSet:
        # Uses the pymeshlab.Mesh constructor directly rather than the OBJ-file
        # workaround in add_mesh_to_meshset. Known to segfault on some platforms
        # (see https://github.com/cnr-isti-vclab/PyMeshLab/issues/392).
        ms = PyMeshLabInterface.create_meshset()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=v, face_matrix=f))
        return ms

    @staticmethod
    def get_mesh_vf(ms: pymeshlab.MeshSet, idx: int = 0):
        m = ms.mesh(idx)
        return m.vertex_matrix(), m.face_matrix()

    @staticmethod
    def get_vertex_scalar_array(ms: pymeshlab.MeshSet, idx: int = 0):
        return ms.mesh(idx).vertex_scalar_array()
