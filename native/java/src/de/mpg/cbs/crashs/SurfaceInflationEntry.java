package de.mpg.cbs.crashs;

import org.graalvm.nativeimage.IsolateThread;
import org.graalvm.nativeimage.ObjectHandle;
import org.graalvm.nativeimage.ObjectHandles;
import org.graalvm.nativeimage.c.function.CEntryPoint;
import org.graalvm.nativeimage.c.type.CCharPointer;
import org.graalvm.nativeimage.c.type.CFloatPointer;
import org.graalvm.nativeimage.c.type.CIntPointer;
import org.graalvm.nativeimage.c.type.CTypeConversion;

import de.mpg.cbs.core.surface.SurfaceInflation;

public final class SurfaceInflationEntry {

    static final class State {
        final SurfaceInflation algo = new SurfaceInflation();
        String lastError = "";
    }

    @CEntryPoint(name = "crashs_inflate_create")
    public static ObjectHandle create(IsolateThread thread) {
        return ObjectHandles.getGlobal().create(new State());
    }

    // weightingMethod: 1=AREA, 2=DIST, 3=NUMV (matches SurfaceInflation.AREA/DIST/NUMV constants)
    @CEntryPoint(name = "crashs_inflate_set_inputs")
    public static int setInputs(IsolateThread thread, ObjectHandle handle,
                                 int numPoints, int numTriangles,
                                 CFloatPointer points, CIntPointer triangles,
                                 float stepSize, int maxIter, float maxCurv,
                                 int weightingMethod, float regularization, float centering) {
        State s = ObjectHandles.getGlobal().get(handle);
        try {
            s.algo.setSurfacePoints(NativeArrayUtil.toFloatArray(points, numPoints * 3));
            s.algo.setSurfaceTriangles(NativeArrayUtil.toIntArray(triangles, numTriangles * 3));
            s.algo.setStepSize(stepSize);
            s.algo.setMaxIter(maxIter);
            s.algo.setMaxCurv(maxCurv);
            s.algo.setWeightingMethod(weightingMethod);
            s.algo.setRegularization(regularization);
            s.algo.setCentering(centering);
            return 0;
        } catch (Throwable t) { s.lastError = NativeArrayUtil.describe(t); return -1; }
    }

    @CEntryPoint(name = "crashs_inflate_execute")
    public static int execute(IsolateThread thread, ObjectHandle handle) {
        State s = ObjectHandles.getGlobal().get(handle);
        try {
            s.algo.execute();
            return 0;
        } catch (Throwable t) { s.lastError = NativeArrayUtil.describe(t); return -1; }
    }

    @CEntryPoint(name = "crashs_inflate_get_points")
    public static int getPoints(IsolateThread thread, ObjectHandle handle, CFloatPointer outPtr, int outLen) {
        State s = ObjectHandles.getGlobal().get(handle);
        try {
            float[] r = s.algo.getInflatedSurfacePoints();
            if (r.length != outLen) { s.lastError = "size mismatch"; return -2; }
            NativeArrayUtil.copyOut(r, outPtr, outLen);
            return 0;
        } catch (Throwable t) { s.lastError = NativeArrayUtil.describe(t); return -1; }
    }

    @CEntryPoint(name = "crashs_inflate_get_triangles")
    public static int getTriangles(IsolateThread thread, ObjectHandle handle, CIntPointer outPtr, int outLen) {
        State s = ObjectHandles.getGlobal().get(handle);
        try {
            int[] r = s.algo.getInflatedSurfaceTriangles();
            if (r.length != outLen) { s.lastError = "size mismatch"; return -2; }
            NativeArrayUtil.copyOut(r, outPtr, outLen);
            return 0;
        } catch (Throwable t) { s.lastError = NativeArrayUtil.describe(t); return -1; }
    }

    @CEntryPoint(name = "crashs_inflate_get_values")
    public static int getValues(IsolateThread thread, ObjectHandle handle, CFloatPointer outPtr, int outLen) {
        State s = ObjectHandles.getGlobal().get(handle);
        try {
            float[] r = s.algo.getInflatedSurfaceValues();
            if (r.length != outLen) { s.lastError = "size mismatch"; return -2; }
            NativeArrayUtil.copyOut(r, outPtr, outLen);
            return 0;
        } catch (Throwable t) { s.lastError = NativeArrayUtil.describe(t); return -1; }
    }

    @CEntryPoint(name = "crashs_inflate_get_last_error")
    public static void getLastError(IsolateThread thread, ObjectHandle handle, CCharPointer buf, int bufLen) {
        State s = ObjectHandles.getGlobal().get(handle);
        CTypeConversion.toCString(s.lastError == null ? "" : s.lastError, buf, org.graalvm.word.WordFactory.unsigned(bufLen));
    }

    @CEntryPoint(name = "crashs_inflate_destroy")
    public static void destroy(IsolateThread thread, ObjectHandle handle) {
        ObjectHandles.getGlobal().destroy(handle);
    }
}
