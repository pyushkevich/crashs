package de.mpg.cbs.crashs;

import org.graalvm.nativeimage.IsolateThread;
import org.graalvm.nativeimage.ObjectHandle;
import org.graalvm.nativeimage.ObjectHandles;
import org.graalvm.nativeimage.c.function.CEntryPoint;
import org.graalvm.nativeimage.c.type.CCharPointer;
import org.graalvm.nativeimage.c.type.CFloatPointer;
import org.graalvm.nativeimage.c.type.CIntPointer;
import org.graalvm.nativeimage.c.type.CTypeConversion;

import de.mpg.cbs.core.laminar.LaminarVolumetricLayering;

public final class VolumetricLayeringEntry {

    static final class State {
        final LaminarVolumetricLayering algo = new LaminarVolumetricLayering();
        String lastError = "";
    }

    @CEntryPoint(name = "crashs_layering_create")
    public static ObjectHandle create(IsolateThread thread) {
        return ObjectHandles.getGlobal().create(new State());
    }

    @CEntryPoint(name = "crashs_layering_set_inputs")
    public static int setInputs(IsolateThread thread, ObjectHandle handle,
                                 int nx, int ny, int nz,
                                 float rx, float ry, float rz,
                                 CFloatPointer innerDistanceImage,
                                 CFloatPointer outerDistanceImage,
                                 int numberOfLayers,
                                 int maxNarrowBandIterations,
                                 float minNarrowBandChange,
                                 CCharPointer layeringMethod,
                                 CCharPointer layeringDirection,
                                 int curvatureApproximationScale,
                                 float ratioSmoothingKernelSize,
                                 boolean presmoothCorticalSurfaces,
                                 CCharPointer topology,
                                 CCharPointer lutDir) {
        State s = ObjectHandles.getGlobal().get(handle);
        try {
            s.algo.setDimensions(nx, ny, nz);
            s.algo.setResolutions(rx, ry, rz);
            int n = nx * ny * nz;
            s.algo.setInnerDistanceImage(NativeArrayUtil.toFloatArray(innerDistanceImage, n));
            s.algo.setOuterDistanceImage(NativeArrayUtil.toFloatArray(outerDistanceImage, n));
            s.algo.setNumberOfLayers(numberOfLayers);
            s.algo.setMaxNarrowBandIterations(maxNarrowBandIterations);
            s.algo.setMinNarrowBandChange(minNarrowBandChange);
            s.algo.setLayeringMethod(CTypeConversion.toJavaString(layeringMethod));
            s.algo.setLayeringDirection(CTypeConversion.toJavaString(layeringDirection));
            s.algo.setCurvatureApproximationScale(curvatureApproximationScale);
            s.algo.setRatioSmoothingKernelSize(ratioSmoothingKernelSize);
            s.algo.setPresmoothCorticalSurfaces(presmoothCorticalSurfaces);
            s.algo.setTopology(CTypeConversion.toJavaString(topology));
            s.algo.setTopologyLUTdirectory(CTypeConversion.toJavaString(lutDir));
            return 0;
        } catch (Throwable t) { s.lastError = NativeArrayUtil.describe(t); return -1; }
    }

    @CEntryPoint(name = "crashs_layering_execute")
    public static int execute(IsolateThread thread, ObjectHandle handle) {
        State s = ObjectHandles.getGlobal().get(handle);
        try {
            s.algo.execute();
            return 0;
        } catch (Throwable t) { s.lastError = NativeArrayUtil.describe(t); return -1; }
    }

    // Fixed-size (nx*ny*nz) outputs
    @CEntryPoint(name = "crashs_layering_get_depth")
    public static int getDepth(IsolateThread thread, ObjectHandle handle, CFloatPointer outPtr, int outLen) {
        State s = ObjectHandles.getGlobal().get(handle);
        try {
            float[] r = s.algo.getContinuousDepthMeasurement();
            if (r.length != outLen) { s.lastError = "size mismatch"; return -2; }
            NativeArrayUtil.copyOut(r, outPtr, outLen);
            return 0;
        } catch (Throwable t) { s.lastError = NativeArrayUtil.describe(t); return -1; }
    }

    @CEntryPoint(name = "crashs_layering_get_discrete_layers")
    public static int getDiscreteLayers(IsolateThread thread, ObjectHandle handle, CIntPointer outPtr, int outLen) {
        State s = ObjectHandles.getGlobal().get(handle);
        try {
            byte[] r = s.algo.getDiscreteSampledLayers();
            if (r.length != outLen) { s.lastError = "size mismatch"; return -2; }
            for (int i = 0; i < outLen; i++) outPtr.write(i, r[i]);
            return 0;
        } catch (Throwable t) { s.lastError = NativeArrayUtil.describe(t); return -1; }
    }

    // Variable-size (nx*ny*nz * (numLayers+1) or numLayers) outputs - query size first
    @CEntryPoint(name = "crashs_layering_get_boundary_surfaces_size")
    public static int getBoundarySurfacesSize(IsolateThread thread, ObjectHandle handle) {
        State s = ObjectHandles.getGlobal().get(handle);
        return s.algo.getLayerBoundarySurfaces().length;
    }

    @CEntryPoint(name = "crashs_layering_get_centered_surfaces_size")
    public static int getCenteredSurfacesSize(IsolateThread thread, ObjectHandle handle) {
        State s = ObjectHandles.getGlobal().get(handle);
        return s.algo.getLayerCenteredSurfaces().length;
    }

    @CEntryPoint(name = "crashs_layering_get_boundary_surfaces")
    public static int getBoundarySurfaces(IsolateThread thread, ObjectHandle handle, CFloatPointer outPtr, int outLen) {
        State s = ObjectHandles.getGlobal().get(handle);
        try {
            float[] r = s.algo.getLayerBoundarySurfaces();
            if (r.length != outLen) { s.lastError = "size mismatch"; return -2; }
            NativeArrayUtil.copyOut(r, outPtr, outLen);
            return 0;
        } catch (Throwable t) { s.lastError = NativeArrayUtil.describe(t); return -1; }
    }

    @CEntryPoint(name = "crashs_layering_get_centered_surfaces")
    public static int getCenteredSurfaces(IsolateThread thread, ObjectHandle handle, CFloatPointer outPtr, int outLen) {
        State s = ObjectHandles.getGlobal().get(handle);
        try {
            float[] r = s.algo.getLayerCenteredSurfaces();
            if (r.length != outLen) { s.lastError = "size mismatch"; return -2; }
            NativeArrayUtil.copyOut(r, outPtr, outLen);
            return 0;
        } catch (Throwable t) { s.lastError = NativeArrayUtil.describe(t); return -1; }
    }

    @CEntryPoint(name = "crashs_layering_get_last_error")
    public static void getLastError(IsolateThread thread, ObjectHandle handle, CCharPointer buf, int bufLen) {
        State s = ObjectHandles.getGlobal().get(handle);
        CTypeConversion.toCString(s.lastError == null ? "" : s.lastError, buf, org.graalvm.word.WordFactory.unsigned(bufLen));
    }

    @CEntryPoint(name = "crashs_layering_destroy")
    public static void destroy(IsolateThread thread, ObjectHandle handle) {
        ObjectHandles.getGlobal().destroy(handle);
    }
}
