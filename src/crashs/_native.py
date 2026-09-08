"""Direct bindings to a GraalVM Native Image build of the cbstools-public algorithms
that crashs needs (topology correction, CRUISE cortical extraction, levelset-to-mesh,
surface inflation, volumetric layering) - replacing the nighres/JCC/JVM dependency.

The compiled shared library has no JVM at runtime: it is the cbstools-public Java code
AOT-compiled once at wheel-build time via GraalVM native-image (see ../native/). This
module talks to it with ctypes and mirrors nighres's numpy-in/numpy-out call signatures
(including its Fortran-order array flattening convention) so callers see the same shapes.
"""
import ctypes
import os
import platform

import numpy as np

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


class NativeAlgorithmError(RuntimeError):
    """Raised when the underlying cbstools native call reports a failure."""


def _lib_path():
    # Deliberately not importlib.resources: under an editable install (scikit-build-core
    # / pip -e .), importlib.resources.files('crashs') resolves to a MultiplexedPath
    # spanning both the source tree and the site-packages stub, and its str() is not a
    # usable filesystem path. This package always ships with a real compiled binary
    # (never a zipped/zipapp install), so a plain __file__-relative path is always valid.
    system = platform.system()
    name = {
        "Linux": "libcbstools_native.so",
        "Darwin": "libcbstools_native.dylib",
        "Windows": "cbstools_native.dll",
    }[system]
    return os.path.join(_PACKAGE_DIR, "_native_lib", name)


def _lut_dir():
    # cbstools-public's CriticalPointLUT concatenates lutdir + filename directly (no
    # separator inserted), so this must end with a path separator.
    return os.path.join(_PACKAGE_DIR, "_native_data", "topology_lut") + os.sep


_lib = None
_thread = None


def _lib_and_thread():
    """Lazily load the shared library and create one GraalVM isolate for the process."""
    global _lib, _thread
    if _lib is not None:
        return _lib, _thread

    lib = ctypes.CDLL(_lib_path())

    lib.graal_create_isolate.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)
    ]
    lib.graal_create_isolate.restype = ctypes.c_int

    isolate = ctypes.c_void_p()
    thread = ctypes.c_void_p()
    rc = lib.graal_create_isolate(None, ctypes.byref(isolate), ctypes.byref(thread))
    if rc != 0:
        raise NativeAlgorithmError(f"graal_create_isolate failed with code {rc}")

    _declare_signatures(lib)
    _lib, _thread = lib, thread
    return _lib, _thread


def _declare_signatures(lib):
    voidp = ctypes.c_void_p
    fptr = ctypes.POINTER(ctypes.c_float)
    iptr = ctypes.POINTER(ctypes.c_int)
    cstr = ctypes.c_char_p

    # -- topology_correction --
    lib.crashs_topocorr_create.argtypes = [voidp]
    lib.crashs_topocorr_create.restype = voidp
    lib.crashs_topocorr_set_inputs.argtypes = [
        voidp, voidp, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, fptr,
        cstr, cstr, cstr, cstr, ctypes.c_float,
    ]
    lib.crashs_topocorr_set_inputs.restype = ctypes.c_int
    lib.crashs_topocorr_execute.argtypes = [voidp, voidp]
    lib.crashs_topocorr_execute.restype = ctypes.c_int
    lib.crashs_topocorr_get_corrected_image.argtypes = [voidp, voidp, fptr, ctypes.c_int]
    lib.crashs_topocorr_get_corrected_image.restype = ctypes.c_int
    lib.crashs_topocorr_get_corrected_object_image.argtypes = [voidp, voidp, iptr, ctypes.c_int]
    lib.crashs_topocorr_get_corrected_object_image.restype = ctypes.c_int
    lib.crashs_topocorr_get_last_error.argtypes = [voidp, voidp, ctypes.c_char_p, ctypes.c_int]
    lib.crashs_topocorr_get_last_error.restype = None
    lib.crashs_topocorr_destroy.argtypes = [voidp, voidp]
    lib.crashs_topocorr_destroy.restype = None

    # -- cruise_cortex_extraction --
    lib.crashs_cruise_create.argtypes = [voidp]
    lib.crashs_cruise_create.restype = voidp
    lib.crashs_cruise_set_inputs.argtypes = [
        voidp, voidp, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
        iptr, fptr, fptr, fptr,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int,
        ctypes.c_bool, ctypes.c_bool, ctypes.c_float, cstr, cstr,
    ]
    lib.crashs_cruise_set_inputs.restype = ctypes.c_int
    lib.crashs_cruise_execute.argtypes = [voidp, voidp]
    lib.crashs_cruise_execute.restype = ctypes.c_int
    lib.crashs_cruise_get_cortex_mask.argtypes = [voidp, voidp, iptr, ctypes.c_int]
    lib.crashs_cruise_get_cortex_mask.restype = ctypes.c_int
    for name in (
        "get_wmgm_levelset", "get_gmcsf_levelset", "get_central_levelset",
        "get_cortical_thickness", "get_cerebral_wm_probability",
        "get_cortical_gm_probability", "get_sulcal_csf_probability",
    ):
        fn = getattr(lib, f"crashs_cruise_{name}")
        fn.argtypes = [voidp, voidp, fptr, ctypes.c_int]
        fn.restype = ctypes.c_int
    lib.crashs_cruise_get_last_error.argtypes = [voidp, voidp, ctypes.c_char_p, ctypes.c_int]
    lib.crashs_cruise_get_last_error.restype = None
    lib.crashs_cruise_destroy.argtypes = [voidp, voidp]
    lib.crashs_cruise_destroy.restype = None

    # -- levelset_to_mesh --
    lib.crashs_l2m_create.argtypes = [voidp]
    lib.crashs_l2m_create.restype = voidp
    lib.crashs_l2m_set_inputs.argtypes = [
        voidp, voidp, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, fptr,
        cstr, ctypes.c_float, ctypes.c_bool,
    ]
    lib.crashs_l2m_set_inputs.restype = ctypes.c_int
    lib.crashs_l2m_execute.argtypes = [voidp, voidp]
    lib.crashs_l2m_execute.restype = ctypes.c_int
    lib.crashs_l2m_get_num_points.argtypes = [voidp, voidp]
    lib.crashs_l2m_get_num_points.restype = ctypes.c_int
    lib.crashs_l2m_get_num_triangles.argtypes = [voidp, voidp]
    lib.crashs_l2m_get_num_triangles.restype = ctypes.c_int
    lib.crashs_l2m_get_points.argtypes = [voidp, voidp, fptr, ctypes.c_int]
    lib.crashs_l2m_get_points.restype = ctypes.c_int
    lib.crashs_l2m_get_triangles.argtypes = [voidp, voidp, iptr, ctypes.c_int]
    lib.crashs_l2m_get_triangles.restype = ctypes.c_int
    lib.crashs_l2m_get_last_error.argtypes = [voidp, voidp, ctypes.c_char_p, ctypes.c_int]
    lib.crashs_l2m_get_last_error.restype = None
    lib.crashs_l2m_destroy.argtypes = [voidp, voidp]
    lib.crashs_l2m_destroy.restype = None

    # -- surface_inflation --
    lib.crashs_inflate_create.argtypes = [voidp]
    lib.crashs_inflate_create.restype = voidp
    lib.crashs_inflate_set_inputs.argtypes = [
        voidp, voidp, ctypes.c_int, ctypes.c_int, fptr, iptr,
        ctypes.c_float, ctypes.c_int, ctypes.c_float,
        ctypes.c_int, ctypes.c_float, ctypes.c_float,
    ]
    lib.crashs_inflate_set_inputs.restype = ctypes.c_int
    lib.crashs_inflate_execute.argtypes = [voidp, voidp]
    lib.crashs_inflate_execute.restype = ctypes.c_int
    lib.crashs_inflate_get_points.argtypes = [voidp, voidp, fptr, ctypes.c_int]
    lib.crashs_inflate_get_points.restype = ctypes.c_int
    lib.crashs_inflate_get_triangles.argtypes = [voidp, voidp, iptr, ctypes.c_int]
    lib.crashs_inflate_get_triangles.restype = ctypes.c_int
    lib.crashs_inflate_get_values.argtypes = [voidp, voidp, fptr, ctypes.c_int]
    lib.crashs_inflate_get_values.restype = ctypes.c_int
    lib.crashs_inflate_get_last_error.argtypes = [voidp, voidp, ctypes.c_char_p, ctypes.c_int]
    lib.crashs_inflate_get_last_error.restype = None
    lib.crashs_inflate_destroy.argtypes = [voidp, voidp]
    lib.crashs_inflate_destroy.restype = None

    # -- volumetric_layering --
    lib.crashs_layering_create.argtypes = [voidp]
    lib.crashs_layering_create.restype = voidp
    lib.crashs_layering_set_inputs.argtypes = [
        voidp, voidp, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, fptr, fptr,
        ctypes.c_int, ctypes.c_int, ctypes.c_float, cstr, cstr,
        ctypes.c_int, ctypes.c_float, ctypes.c_bool, cstr, cstr,
    ]
    lib.crashs_layering_set_inputs.restype = ctypes.c_int
    lib.crashs_layering_execute.argtypes = [voidp, voidp]
    lib.crashs_layering_execute.restype = ctypes.c_int
    lib.crashs_layering_get_depth.argtypes = [voidp, voidp, fptr, ctypes.c_int]
    lib.crashs_layering_get_depth.restype = ctypes.c_int
    lib.crashs_layering_get_discrete_layers.argtypes = [voidp, voidp, iptr, ctypes.c_int]
    lib.crashs_layering_get_discrete_layers.restype = ctypes.c_int
    lib.crashs_layering_get_boundary_surfaces_size.argtypes = [voidp, voidp]
    lib.crashs_layering_get_boundary_surfaces_size.restype = ctypes.c_int
    lib.crashs_layering_get_centered_surfaces_size.argtypes = [voidp, voidp]
    lib.crashs_layering_get_centered_surfaces_size.restype = ctypes.c_int
    lib.crashs_layering_get_boundary_surfaces.argtypes = [voidp, voidp, fptr, ctypes.c_int]
    lib.crashs_layering_get_boundary_surfaces.restype = ctypes.c_int
    lib.crashs_layering_get_centered_surfaces.argtypes = [voidp, voidp, fptr, ctypes.c_int]
    lib.crashs_layering_get_centered_surfaces.restype = ctypes.c_int
    lib.crashs_layering_get_last_error.argtypes = [voidp, voidp, ctypes.c_char_p, ctypes.c_int]
    lib.crashs_layering_get_last_error.restype = None
    lib.crashs_layering_destroy.argtypes = [voidp, voidp]
    lib.crashs_layering_destroy.restype = None


def _raise_last_error(lib, thread, handle, get_last_error_fn, context):
    buf = ctypes.create_string_buffer(8192)
    get_last_error_fn(thread, handle, buf, 8192)
    msg = buf.value.decode(errors="replace")
    raise NativeAlgorithmError(f"{context} failed:\n{msg}")


def _f32(a):
    return np.asarray(a, dtype=np.float32).flatten(order="F")


def _fptr(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _iptr(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))


def topology_correction(data, resolution, shape_type,
                         connectivity="wcs", propagation="object->background",
                         minimum_distance=1e-5):
    """Equivalent to nighres.shape.topology_correction, operating on numpy arrays
    (rather than nifti file paths) and returning {'corrected': float32, 'object': int32},
    both shaped like `data` (order matches input, internally Fortran-flattened like nighres)."""
    lib, thread = _lib_and_thread()
    nx, ny, nz = data.shape
    flat = _f32(data)
    handle = lib.crashs_topocorr_create(thread)
    try:
        rc = lib.crashs_topocorr_set_inputs(
            thread, handle, nx, ny, nz,
            resolution[0], resolution[1], resolution[2],
            _fptr(flat),
            shape_type.encode(), connectivity.encode(),
            _lut_dir().encode(), propagation.encode(),
            minimum_distance,
        )
        if rc != 0:
            _raise_last_error(lib, thread, handle, lib.crashs_topocorr_get_last_error, "topology_correction.set_inputs")
        rc = lib.crashs_topocorr_execute(thread, handle)
        if rc != 0:
            _raise_last_error(lib, thread, handle, lib.crashs_topocorr_get_last_error, "topology_correction.execute")

        n = nx * ny * nz
        corrected = np.empty(n, dtype=np.float32)
        rc = lib.crashs_topocorr_get_corrected_image(thread, handle, _fptr(corrected), n)
        if rc != 0:
            _raise_last_error(lib, thread, handle, lib.crashs_topocorr_get_last_error, "topology_correction.get_corrected_image")
        obj = np.empty(n, dtype=np.int32)
        rc = lib.crashs_topocorr_get_corrected_object_image(thread, handle, _iptr(obj), n)
        if rc != 0:
            _raise_last_error(lib, thread, handle, lib.crashs_topocorr_get_last_error, "topology_correction.get_corrected_object_image")

        return {
            "corrected": corrected.reshape((nx, ny, nz), order="F"),
            "object": obj.reshape((nx, ny, nz), order="F"),
        }
    finally:
        lib.crashs_topocorr_destroy(thread, handle)


def cruise_cortex_extraction(init_image, wm_image, gm_image, csf_image, resolution,
                              data_weight=0.4, regularization_weight=0.1,
                              max_iterations=500, normalize_probabilities=False,
                              correct_wm_pv=True, wm_dropoff_dist=1.0,
                              topology="wcs"):
    """Equivalent to nighres.cortex.cruise_cortex_extraction on in-memory numpy arrays.

    Note: cbstools's CortexOptimCRUISE also exposes an edge-weight parameter
    (setEdgeWeight/setSignedEdgemapImage), but nighres's wrapper never sets it either
    (leaving the Java-side default of 0.0, which disables that code path entirely) -
    matched here for parity rather than exposed, since without an edge map input it
    would otherwise hit a null signed-edge-map array inside Mp2rageCortexGdm."""
    edge_weight = 0.0
    lib, thread = _lib_and_thread()
    nx, ny, nz = init_image.shape
    n = nx * ny * nz
    init_flat = np.asarray(init_image, dtype=np.int32).flatten(order="F")
    wm_flat = _f32(wm_image)
    gm_flat = _f32(gm_image)
    csf_flat = _f32(csf_image)

    handle = lib.crashs_cruise_create(thread)
    try:
        rc = lib.crashs_cruise_set_inputs(
            thread, handle, nx, ny, nz,
            resolution[0], resolution[1], resolution[2],
            _iptr(init_flat), _fptr(wm_flat), _fptr(gm_flat), _fptr(csf_flat),
            data_weight, edge_weight, regularization_weight, max_iterations,
            normalize_probabilities, correct_wm_pv, wm_dropoff_dist,
            topology.encode(), _lut_dir().encode(),
        )
        if rc != 0:
            _raise_last_error(lib, thread, handle, lib.crashs_cruise_get_last_error, "cruise_cortex_extraction.set_inputs")
        rc = lib.crashs_cruise_execute(thread, handle)
        if rc != 0:
            _raise_last_error(lib, thread, handle, lib.crashs_cruise_get_last_error, "cruise_cortex_extraction.execute")

        def fetch_float(fn):
            out = np.empty(n, dtype=np.float32)
            rc = fn(thread, handle, _fptr(out), n)
            if rc != 0:
                _raise_last_error(lib, thread, handle, lib.crashs_cruise_get_last_error, fn.__name__)
            return out.reshape((nx, ny, nz), order="F")

        cortex = np.empty(n, dtype=np.int32)
        rc = lib.crashs_cruise_get_cortex_mask(thread, handle, _iptr(cortex), n)
        if rc != 0:
            _raise_last_error(lib, thread, handle, lib.crashs_cruise_get_last_error, "get_cortex_mask")

        return {
            "cortex": cortex.reshape((nx, ny, nz), order="F"),
            "gwb": fetch_float(lib.crashs_cruise_get_wmgm_levelset),
            "cgb": fetch_float(lib.crashs_cruise_get_gmcsf_levelset),
            "avg": fetch_float(lib.crashs_cruise_get_central_levelset),
            "thickness": fetch_float(lib.crashs_cruise_get_cortical_thickness),
            "pwm": fetch_float(lib.crashs_cruise_get_cerebral_wm_probability),
            "pgm": fetch_float(lib.crashs_cruise_get_cortical_gm_probability),
            "pcsf": fetch_float(lib.crashs_cruise_get_sulcal_csf_probability),
        }
    finally:
        lib.crashs_cruise_destroy(thread, handle)


def levelset_to_mesh(levelset_image, resolution, connectivity="18/6", level=0.0, inclusive=True):
    """Equivalent to nighres.surface.levelset_to_mesh. Returns {'points': (N,3) float32,
    'faces': (M,3) int32}."""
    lib, thread = _lib_and_thread()
    nx, ny, nz = levelset_image.shape
    flat = _f32(levelset_image)

    handle = lib.crashs_l2m_create(thread)
    try:
        rc = lib.crashs_l2m_set_inputs(
            thread, handle, nx, ny, nz,
            resolution[0], resolution[1], resolution[2],
            _fptr(flat), connectivity.encode(), level, inclusive,
        )
        if rc != 0:
            _raise_last_error(lib, thread, handle, lib.crashs_l2m_get_last_error, "levelset_to_mesh.set_inputs")
        rc = lib.crashs_l2m_execute(thread, handle)
        if rc != 0:
            _raise_last_error(lib, thread, handle, lib.crashs_l2m_get_last_error, "levelset_to_mesh.execute")

        npts = lib.crashs_l2m_get_num_points(thread, handle)
        ntri = lib.crashs_l2m_get_num_triangles(thread, handle)

        points = np.empty(npts * 3, dtype=np.float32)
        rc = lib.crashs_l2m_get_points(thread, handle, _fptr(points), npts * 3)
        if rc != 0:
            _raise_last_error(lib, thread, handle, lib.crashs_l2m_get_last_error, "levelset_to_mesh.get_points")
        triangles = np.empty(ntri * 3, dtype=np.int32)
        rc = lib.crashs_l2m_get_triangles(thread, handle, _iptr(triangles), ntri * 3)
        if rc != 0:
            _raise_last_error(lib, thread, handle, lib.crashs_l2m_get_last_error, "levelset_to_mesh.get_triangles")

        return {
            "points": points.reshape((npts, 3)),
            "faces": triangles.reshape((ntri, 3)),
        }
    finally:
        lib.crashs_l2m_destroy(thread, handle)


# weightingMethod constants, matching SurfaceInflation.AREA/DIST/NUMV in cbstools-public
_INFLATE_WEIGHTING = {"area": 1, "dist": 2, "numv": 3}


def surface_inflation(points, faces, step_size=0.75, max_iter=2000, max_curv=10.0,
                       method="area", regularization=1.0, centering=True):
    """Equivalent to nighres.surface.surface_inflation. `points`/`faces` are (N,3)/(M,3)
    arrays as returned by levelset_to_mesh. Returns {'points', 'faces', 'values'}."""
    lib, thread = _lib_and_thread()
    npts = points.shape[0]
    ntri = faces.shape[0]
    points_flat = np.asarray(points, dtype=np.float32).flatten()
    faces_flat = np.asarray(faces, dtype=np.int32).flatten()

    handle = lib.crashs_inflate_create(thread)
    try:
        rc = lib.crashs_inflate_set_inputs(
            thread, handle, npts, ntri,
            _fptr(points_flat), _iptr(faces_flat),
            step_size, max_iter, max_curv,
            _INFLATE_WEIGHTING[method], regularization, float(centering),
        )
        if rc != 0:
            _raise_last_error(lib, thread, handle, lib.crashs_inflate_get_last_error, "surface_inflation.set_inputs")
        rc = lib.crashs_inflate_execute(thread, handle)
        if rc != 0:
            _raise_last_error(lib, thread, handle, lib.crashs_inflate_get_last_error, "surface_inflation.execute")

        out_points = np.empty(npts * 3, dtype=np.float32)
        rc = lib.crashs_inflate_get_points(thread, handle, _fptr(out_points), npts * 3)
        if rc != 0:
            _raise_last_error(lib, thread, handle, lib.crashs_inflate_get_last_error, "surface_inflation.get_points")
        out_faces = np.empty(ntri * 3, dtype=np.int32)
        rc = lib.crashs_inflate_get_triangles(thread, handle, _iptr(out_faces), ntri * 3)
        if rc != 0:
            _raise_last_error(lib, thread, handle, lib.crashs_inflate_get_last_error, "surface_inflation.get_triangles")
        out_values = np.empty(npts, dtype=np.float32)
        rc = lib.crashs_inflate_get_values(thread, handle, _fptr(out_values), npts)
        if rc != 0:
            _raise_last_error(lib, thread, handle, lib.crashs_inflate_get_last_error, "surface_inflation.get_values")

        return {
            "points": out_points.reshape((npts, 3)),
            "faces": out_faces.reshape((ntri, 3)),
            "values": out_values,
        }
    finally:
        lib.crashs_inflate_destroy(thread, handle)


def volumetric_layering(inner_levelset, outer_levelset, resolution, n_layers=10,
                         max_narrow_band_iterations=100, min_narrow_band_change=0.001,
                         method="volume-preserving", direction="outward",
                         curvature_approx_scale=3, ratio_smoothing_kernel=1.0,
                         presmooth_cortical_surfaces=True, topology="wcs"):
    """Equivalent to nighres.laminar.volumetric_layering. Returns {'depth', 'labels',
    'boundaries', 'midlayers'} where 'boundaries'/'midlayers' are shaped
    (nx, ny, nz, n_layers+1)/(nx, ny, nz, n_layers)."""
    lib, thread = _lib_and_thread()
    nx, ny, nz = inner_levelset.shape
    n = nx * ny * nz
    inner_flat = _f32(inner_levelset)
    outer_flat = _f32(outer_levelset)

    handle = lib.crashs_layering_create(thread)
    try:
        rc = lib.crashs_layering_set_inputs(
            thread, handle, nx, ny, nz,
            resolution[0], resolution[1], resolution[2],
            _fptr(inner_flat), _fptr(outer_flat),
            n_layers, max_narrow_band_iterations, min_narrow_band_change,
            method.encode(), direction.encode(),
            curvature_approx_scale, ratio_smoothing_kernel, presmooth_cortical_surfaces,
            topology.encode(), _lut_dir().encode(),
        )
        if rc != 0:
            _raise_last_error(lib, thread, handle, lib.crashs_layering_get_last_error, "volumetric_layering.set_inputs")
        rc = lib.crashs_layering_execute(thread, handle)
        if rc != 0:
            _raise_last_error(lib, thread, handle, lib.crashs_layering_get_last_error, "volumetric_layering.execute")

        depth = np.empty(n, dtype=np.float32)
        rc = lib.crashs_layering_get_depth(thread, handle, _fptr(depth), n)
        if rc != 0:
            _raise_last_error(lib, thread, handle, lib.crashs_layering_get_last_error, "get_depth")

        labels = np.empty(n, dtype=np.int32)
        rc = lib.crashs_layering_get_discrete_layers(thread, handle, _iptr(labels), n)
        if rc != 0:
            _raise_last_error(lib, thread, handle, lib.crashs_layering_get_last_error, "get_discrete_layers")

        n_bound = lib.crashs_layering_get_boundary_surfaces_size(thread, handle)
        boundaries = np.empty(n_bound, dtype=np.float32)
        rc = lib.crashs_layering_get_boundary_surfaces(thread, handle, _fptr(boundaries), n_bound)
        if rc != 0:
            _raise_last_error(lib, thread, handle, lib.crashs_layering_get_last_error, "get_boundary_surfaces")

        n_mid = lib.crashs_layering_get_centered_surfaces_size(thread, handle)
        midlayers = np.empty(n_mid, dtype=np.float32)
        rc = lib.crashs_layering_get_centered_surfaces(thread, handle, _fptr(midlayers), n_mid)
        if rc != 0:
            _raise_last_error(lib, thread, handle, lib.crashs_layering_get_last_error, "get_centered_surfaces")

        return {
            "depth": depth.reshape((nx, ny, nz), order="F"),
            "labels": labels.reshape((nx, ny, nz), order="F"),
            "boundaries": boundaries.reshape((nx, ny, nz, n_bound // n), order="F"),
            "midlayers": midlayers.reshape((nx, ny, nz, n_mid // n), order="F"),
        }
    finally:
        lib.crashs_layering_destroy(thread, handle)
