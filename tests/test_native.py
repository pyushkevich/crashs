"""Smoke tests for the GraalVM-native cbstools bindings in crashs._native.

These run the actual compiled algorithms (topology correction, marching-cubes
levelset-to-mesh, surface inflation, volumetric layering, CRUISE cortex extraction)
on small synthetic volumes/meshes, exercising the full ctypes/isolate/ObjectHandles
round trip end to end. They require native/build/out/libcbstools_native.* to exist
(built by native/scripts/build_native.sh) and to be importable as crashs._native,
i.e. either an installed wheel or the artifact staged under src/crashs/_native_lib/
and src/crashs/_native_data/ during local development.
"""
import numpy as np
import pytest

from crashs import _native


def _synthetic_binary_object(n=16):
    data = np.zeros((n, n, n), dtype=np.float32)
    data[4:12, 4:12, 4:12] = 1.0
    # a thin tunnel through the middle, to give topology correction something to fix
    data[7:9, 7:9, :] = 1.0
    return data


def test_topology_correction():
    data = _synthetic_binary_object()
    result = _native.topology_correction(
        data, (1.0, 1.0, 1.0), shape_type="binary_object",
        propagation="background->object")
    assert result["corrected"].shape == data.shape
    assert result["object"].shape == data.shape
    assert set(np.unique(result["object"]).tolist()) <= {0, 1}
    # Topology correction on a solid cube with a thin tunnel should remove some
    # foreground voxels (closing the small handle) but not the whole object.
    assert 0 < result["object"].sum() < data.sum()


def test_levelset_to_mesh_and_inflation():
    data = _synthetic_binary_object()
    corr = _native.topology_correction(
        data, (1.0, 1.0, 1.0), shape_type="binary_object",
        propagation="background->object")

    mesh = _native.levelset_to_mesh(corr["corrected"], (1.0, 1.0, 1.0))
    assert mesh["points"].ndim == 2 and mesh["points"].shape[1] == 3
    assert mesh["faces"].ndim == 2 and mesh["faces"].shape[1] == 3
    assert mesh["points"].shape[0] > 0
    assert mesh["faces"].shape[0] > 0

    inflated = _native.surface_inflation(mesh["points"], mesh["faces"], max_iter=10)
    assert inflated["points"].shape == mesh["points"].shape
    assert inflated["faces"].shape == mesh["faces"].shape
    assert inflated["values"].shape == (mesh["points"].shape[0],)


def test_volumetric_layering():
    n = 16
    inner = np.full((n, n, n), -2.0, dtype=np.float32)
    inner[6:10, 6:10, 6:10] = 2.0
    outer = np.full((n, n, n), -4.0, dtype=np.float32)
    outer[4:12, 4:12, 4:12] = 4.0

    layers = _native.volumetric_layering(inner, outer, (1.0, 1.0, 1.0), n_layers=4)
    assert layers["depth"].shape == (n, n, n)
    assert layers["labels"].shape == (n, n, n)
    assert layers["boundaries"].shape == (n, n, n, 5)
    assert layers["midlayers"].shape == (n, n, n, 4)


def test_cruise_cortex_extraction():
    n = 20
    init = np.zeros((n, n, n), dtype=np.int32)
    init[6:14, 6:14, 6:14] = 1
    wm = init.astype(np.float32)
    gm = np.zeros((n, n, n), dtype=np.float32)
    gm[4:16, 4:16, 4:16] = 1.0 - wm[4:16, 4:16, 4:16]
    csf = np.clip(1.0 - wm - gm, 0, 1)

    out = _native.cruise_cortex_extraction(
        init, wm, gm, csf, (1.0, 1.0, 1.0),
        normalize_probabilities=True, max_iterations=20)

    for key in ("cortex", "gwb", "cgb", "avg", "thickness", "pwm", "pgm", "pcsf"):
        assert out[key].shape == (n, n, n), key


def test_error_reporting_on_bad_input():
    # Exercise the status-code/last-error plumbing directly: every getter checks the
    # requested output length against the algorithm's actual result length and returns
    # a non-zero status (without touching the output buffer) on mismatch, since Java
    # exceptions can't cross the native boundary - this is what every _native.* wrapper
    # function relies on to turn native failures into NativeAlgorithmError. A size lie
    # here is a safe way to trigger that path deterministically (no out-of-bounds write:
    # the Java-side length check happens before any copy into the caller's buffer).
    import ctypes
    lib, thread = _native._lib_and_thread()
    data = _synthetic_binary_object()
    nx, ny, nz = data.shape
    flat = _native._f32(data)
    handle = lib.crashs_topocorr_create(thread)
    try:
        rc = lib.crashs_topocorr_set_inputs(
            thread, handle, nx, ny, nz, 1.0, 1.0, 1.0, _native._fptr(flat),
            b"binary_object", b"wcs", _native._lut_dir().encode(),
            b"background->object", 1e-5)
        assert rc == 0
        assert lib.crashs_topocorr_execute(thread, handle) == 0

        wrong_len = nx * ny * nz - 1
        out = np.empty(wrong_len, dtype=np.float32)
        rc = lib.crashs_topocorr_get_corrected_image(thread, handle, _native._fptr(out), wrong_len)
        assert rc != 0

        with pytest.raises(_native.NativeAlgorithmError):
            _native._raise_last_error(lib, thread, handle, lib.crashs_topocorr_get_last_error, "test")
    finally:
        lib.crashs_topocorr_destroy(thread, handle)
