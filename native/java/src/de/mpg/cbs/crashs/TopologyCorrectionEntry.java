package de.mpg.cbs.crashs;

import org.graalvm.nativeimage.IsolateThread;
import org.graalvm.nativeimage.ObjectHandle;
import org.graalvm.nativeimage.ObjectHandles;
import org.graalvm.nativeimage.c.function.CEntryPoint;
import org.graalvm.nativeimage.c.type.CCharPointer;
import org.graalvm.nativeimage.c.type.CFloatPointer;
import org.graalvm.nativeimage.c.type.CIntPointer;
import org.graalvm.nativeimage.c.type.CTypeConversion;

import de.mpg.cbs.core.shape.ShapeTopologyCorrection2;

public final class TopologyCorrectionEntry {

    static final class State {
        final ShapeTopologyCorrection2 algo = new ShapeTopologyCorrection2();
        String lastError = "";
        int nx, ny, nz;
    }

    @CEntryPoint(name = "crashs_topocorr_create")
    public static ObjectHandle create(IsolateThread thread) {
        return ObjectHandles.getGlobal().create(new State());
    }

    @CEntryPoint(name = "crashs_topocorr_set_inputs")
    public static int setInputs(IsolateThread thread, ObjectHandle handle,
                                 int nx, int ny, int nz,
                                 float rx, float ry, float rz,
                                 CFloatPointer shapeData,
                                 CCharPointer shapeType,
                                 CCharPointer topology,
                                 CCharPointer lutDir,
                                 CCharPointer propagation,
                                 float minDistance) {
        State s = ObjectHandles.getGlobal().get(handle);
        try {
            s.nx = nx; s.ny = ny; s.nz = nz;
            s.algo.setDimensions(nx, ny, nz);
            s.algo.setResolutions(rx, ry, rz);

            int n = nx * ny * nz;
            float[] data = new float[n];
            for (int i = 0; i < n; i++) data[i] = shapeData.read(i);
            s.algo.setShapeImage(data);

            s.algo.setShapeImageType(CTypeConversion.toJavaString(shapeType));
            s.algo.setTopology(CTypeConversion.toJavaString(topology));
            s.algo.setTopologyLUTdirectory(CTypeConversion.toJavaString(lutDir));
            s.algo.setPropagationDirection(CTypeConversion.toJavaString(propagation));
            s.algo.setMinimumDistance(minDistance);
            return 0;
        } catch (Throwable t) {
            s.lastError = describe(t);
            return -1;
        }
    }

    @CEntryPoint(name = "crashs_topocorr_execute")
    public static int execute(IsolateThread thread, ObjectHandle handle) {
        State s = ObjectHandles.getGlobal().get(handle);
        try {
            s.algo.execute();
            return 0;
        } catch (Throwable t) {
            s.lastError = describe(t);
            return -1;
        }
    }

    @CEntryPoint(name = "crashs_topocorr_get_corrected_image")
    public static int getCorrectedImage(IsolateThread thread, ObjectHandle handle, CFloatPointer outPtr, int outLen) {
        State s = ObjectHandles.getGlobal().get(handle);
        try {
            float[] result = s.algo.getCorrectedImage();
            if (result.length != outLen) { s.lastError = "size mismatch"; return -2; }
            for (int i = 0; i < outLen; i++) outPtr.write(i, result[i]);
            return 0;
        } catch (Throwable t) {
            s.lastError = describe(t);
            return -1;
        }
    }

    @CEntryPoint(name = "crashs_topocorr_get_corrected_object_image")
    public static int getCorrectedObjectImage(IsolateThread thread, ObjectHandle handle, CIntPointer outPtr, int outLen) {
        State s = ObjectHandles.getGlobal().get(handle);
        try {
            int[] result = s.algo.getCorrectedObjectImage();
            if (result.length != outLen) { s.lastError = "size mismatch"; return -2; }
            for (int i = 0; i < outLen; i++) outPtr.write(i, result[i]);
            return 0;
        } catch (Throwable t) {
            s.lastError = describe(t);
            return -1;
        }
    }

    @CEntryPoint(name = "crashs_topocorr_get_last_error")
    public static void getLastError(IsolateThread thread, ObjectHandle handle, CCharPointer buf, int bufLen) {
        State s = ObjectHandles.getGlobal().get(handle);
        CTypeConversion.toCString(s.lastError == null ? "" : s.lastError, buf, org.graalvm.word.WordFactory.unsigned(bufLen));
    }

    @CEntryPoint(name = "crashs_topocorr_destroy")
    public static void destroy(IsolateThread thread, ObjectHandle handle) {
        ObjectHandles.getGlobal().destroy(handle);
    }

    private static String describe(Throwable t) {
        java.io.StringWriter sw = new java.io.StringWriter();
        t.printStackTrace(new java.io.PrintWriter(sw));
        return sw.toString();
    }
}
