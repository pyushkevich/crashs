package de.mpg.cbs.crashs;

import org.graalvm.nativeimage.IsolateThread;
import org.graalvm.nativeimage.ObjectHandle;
import org.graalvm.nativeimage.ObjectHandles;
import org.graalvm.nativeimage.c.function.CEntryPoint;
import org.graalvm.nativeimage.c.type.CCharPointer;
import org.graalvm.nativeimage.c.type.CFloatPointer;
import org.graalvm.nativeimage.c.type.CIntPointer;
import org.graalvm.nativeimage.c.type.CTypeConversion;

import de.mpg.cbs.core.surface.SurfaceLevelsetToMesh;

public final class LevelsetToMeshEntry {

    static final class State {
        final SurfaceLevelsetToMesh algo = new SurfaceLevelsetToMesh();
        String lastError = "";
    }

    @CEntryPoint(name = "crashs_l2m_create")
    public static ObjectHandle create(IsolateThread thread) {
        return ObjectHandles.getGlobal().create(new State());
    }

    @CEntryPoint(name = "crashs_l2m_set_inputs")
    public static int setInputs(IsolateThread thread, ObjectHandle handle,
                                 int nx, int ny, int nz,
                                 float rx, float ry, float rz,
                                 CFloatPointer levelsetImage,
                                 CCharPointer connectivity,
                                 float zeroLevel,
                                 boolean inclusive) {
        State s = ObjectHandles.getGlobal().get(handle);
        try {
            s.algo.setDimensions(nx, ny, nz);
            s.algo.setResolutions(rx, ry, rz);
            s.algo.setLevelsetImage(NativeArrayUtil.toFloatArray(levelsetImage, nx * ny * nz));
            s.algo.setConnectivity(CTypeConversion.toJavaString(connectivity));
            s.algo.setZeroLevel(zeroLevel);
            s.algo.setInclusive(inclusive);
            return 0;
        } catch (Throwable t) { s.lastError = NativeArrayUtil.describe(t); return -1; }
    }

    @CEntryPoint(name = "crashs_l2m_execute")
    public static int execute(IsolateThread thread, ObjectHandle handle) {
        State s = ObjectHandles.getGlobal().get(handle);
        try {
            s.algo.execute();
            return 0;
        } catch (Throwable t) { s.lastError = NativeArrayUtil.describe(t); return -1; }
    }

    @CEntryPoint(name = "crashs_l2m_get_num_points")
    public static int getNumPoints(IsolateThread thread, ObjectHandle handle) {
        State s = ObjectHandles.getGlobal().get(handle);
        return s.algo.getPointList().length / 3;
    }

    @CEntryPoint(name = "crashs_l2m_get_num_triangles")
    public static int getNumTriangles(IsolateThread thread, ObjectHandle handle) {
        State s = ObjectHandles.getGlobal().get(handle);
        return s.algo.getTriangleList().length / 3;
    }

    @CEntryPoint(name = "crashs_l2m_get_points")
    public static int getPoints(IsolateThread thread, ObjectHandle handle, CFloatPointer outPtr, int outLen) {
        State s = ObjectHandles.getGlobal().get(handle);
        try {
            float[] r = s.algo.getPointList();
            if (r.length != outLen) { s.lastError = "size mismatch"; return -2; }
            NativeArrayUtil.copyOut(r, outPtr, outLen);
            return 0;
        } catch (Throwable t) { s.lastError = NativeArrayUtil.describe(t); return -1; }
    }

    @CEntryPoint(name = "crashs_l2m_get_triangles")
    public static int getTriangles(IsolateThread thread, ObjectHandle handle, CIntPointer outPtr, int outLen) {
        State s = ObjectHandles.getGlobal().get(handle);
        try {
            int[] r = s.algo.getTriangleList();
            if (r.length != outLen) { s.lastError = "size mismatch"; return -2; }
            NativeArrayUtil.copyOut(r, outPtr, outLen);
            return 0;
        } catch (Throwable t) { s.lastError = NativeArrayUtil.describe(t); return -1; }
    }

    @CEntryPoint(name = "crashs_l2m_get_last_error")
    public static void getLastError(IsolateThread thread, ObjectHandle handle, CCharPointer buf, int bufLen) {
        State s = ObjectHandles.getGlobal().get(handle);
        CTypeConversion.toCString(s.lastError == null ? "" : s.lastError, buf, org.graalvm.word.WordFactory.unsigned(bufLen));
    }

    @CEntryPoint(name = "crashs_l2m_destroy")
    public static void destroy(IsolateThread thread, ObjectHandle handle) {
        ObjectHandles.getGlobal().destroy(handle);
    }
}
