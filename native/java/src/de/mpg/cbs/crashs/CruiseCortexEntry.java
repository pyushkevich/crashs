package de.mpg.cbs.crashs;

import org.graalvm.nativeimage.IsolateThread;
import org.graalvm.nativeimage.ObjectHandle;
import org.graalvm.nativeimage.ObjectHandles;
import org.graalvm.nativeimage.c.function.CEntryPoint;
import org.graalvm.nativeimage.c.type.CCharPointer;
import org.graalvm.nativeimage.c.type.CFloatPointer;
import org.graalvm.nativeimage.c.type.CIntPointer;
import org.graalvm.nativeimage.c.type.CTypeConversion;

import de.mpg.cbs.core.cortex.CortexOptimCRUISE;

public final class CruiseCortexEntry {

    static final class State {
        final CortexOptimCRUISE algo = new CortexOptimCRUISE();
        String lastError = "";
    }

    @CEntryPoint(name = "crashs_cruise_create")
    public static ObjectHandle create(IsolateThread thread) {
        return ObjectHandles.getGlobal().create(new State());
    }

    @CEntryPoint(name = "crashs_cruise_set_inputs")
    public static int setInputs(IsolateThread thread, ObjectHandle handle,
                                 int nx, int ny, int nz,
                                 float rx, float ry, float rz,
                                 CIntPointer initImage,
                                 CFloatPointer wmImage,
                                 CFloatPointer gmImage,
                                 CFloatPointer csfImage,
                                 float dataWeight, float edgeWeight, float regularizationWeight,
                                 int maxIterations,
                                 boolean normalizeProbabilities, boolean correctForWMGMpartialVoluming,
                                 float wmDropoffDistance,
                                 CCharPointer topology, CCharPointer lutDir) {
        State s = ObjectHandles.getGlobal().get(handle);
        try {
            s.algo.setDimensions(nx, ny, nz);
            s.algo.setResolutions(rx, ry, rz);

            int n = nx * ny * nz;
            byte[] init = new byte[n];
            for (int i = 0; i < n; i++) init[i] = (byte) initImage.read(i);
            s.algo.setInitialWMSegmentationImage(init);

            s.algo.setFilledWMProbabilityImage(NativeArrayUtil.toFloatArray(wmImage, n));
            s.algo.setGMProbabilityImage(NativeArrayUtil.toFloatArray(gmImage, n));
            s.algo.setCSFandBGProbabilityImage(NativeArrayUtil.toFloatArray(csfImage, n));

            s.algo.setDataWeight(dataWeight);
            s.algo.setEdgeWeight(edgeWeight);
            s.algo.setRegularizationWeight(regularizationWeight);
            s.algo.setMaxIterations(maxIterations);
            s.algo.setNormalizeProbabilities(normalizeProbabilities);
            s.algo.setCorrectForWMGMpartialVoluming(correctForWMGMpartialVoluming);
            s.algo.setWMdropoffDistance(wmDropoffDistance);
            s.algo.setTopology(CTypeConversion.toJavaString(topology));
            s.algo.setTopologyLUTdirectory(CTypeConversion.toJavaString(lutDir));
            return 0;
        } catch (Throwable t) {
            s.lastError = NativeArrayUtil.describe(t);
            return -1;
        }
    }

    @CEntryPoint(name = "crashs_cruise_execute")
    public static int execute(IsolateThread thread, ObjectHandle handle) {
        State s = ObjectHandles.getGlobal().get(handle);
        try {
            s.algo.execute();
            return 0;
        } catch (Throwable t) {
            s.lastError = NativeArrayUtil.describe(t);
            return -1;
        }
    }

    @CEntryPoint(name = "crashs_cruise_get_cortex_mask")
    public static int getCortexMask(IsolateThread thread, ObjectHandle handle, CIntPointer outPtr, int outLen) {
        State s = ObjectHandles.getGlobal().get(handle);
        try {
            byte[] r = s.algo.getCortexMask();
            if (r.length != outLen) { s.lastError = "size mismatch"; return -2; }
            for (int i = 0; i < outLen; i++) outPtr.write(i, r[i]);
            return 0;
        } catch (Throwable t) { s.lastError = NativeArrayUtil.describe(t); return -1; }
    }

    @CEntryPoint(name = "crashs_cruise_get_wmgm_levelset")
    public static int getWMGMLevelset(IsolateThread thread, ObjectHandle handle, CFloatPointer outPtr, int outLen) {
        return copyFloatOut(handle, outPtr, outLen, 1);
    }

    @CEntryPoint(name = "crashs_cruise_get_gmcsf_levelset")
    public static int getGMCSFLevelset(IsolateThread thread, ObjectHandle handle, CFloatPointer outPtr, int outLen) {
        return copyFloatOut(handle, outPtr, outLen, 2);
    }

    @CEntryPoint(name = "crashs_cruise_get_central_levelset")
    public static int getCentralLevelset(IsolateThread thread, ObjectHandle handle, CFloatPointer outPtr, int outLen) {
        return copyFloatOut(handle, outPtr, outLen, 3);
    }

    @CEntryPoint(name = "crashs_cruise_get_cortical_thickness")
    public static int getCorticalThickness(IsolateThread thread, ObjectHandle handle, CFloatPointer outPtr, int outLen) {
        return copyFloatOut(handle, outPtr, outLen, 4);
    }

    @CEntryPoint(name = "crashs_cruise_get_cerebral_wm_probability")
    public static int getCerebralWMprobability(IsolateThread thread, ObjectHandle handle, CFloatPointer outPtr, int outLen) {
        return copyFloatOut(handle, outPtr, outLen, 5);
    }

    @CEntryPoint(name = "crashs_cruise_get_cortical_gm_probability")
    public static int getCorticalGMprobability(IsolateThread thread, ObjectHandle handle, CFloatPointer outPtr, int outLen) {
        return copyFloatOut(handle, outPtr, outLen, 6);
    }

    @CEntryPoint(name = "crashs_cruise_get_sulcal_csf_probability")
    public static int getSulcalCSFprobability(IsolateThread thread, ObjectHandle handle, CFloatPointer outPtr, int outLen) {
        return copyFloatOut(handle, outPtr, outLen, 7);
    }

    // handle is passed via ObjectHandles - dispatch on `which` to avoid 7 near-identical bodies
    private static int copyFloatOut(ObjectHandle handle, CFloatPointer outPtr, int outLen, int which) {
        State s = ObjectHandles.getGlobal().get(handle);
        try {
            float[] r;
            switch (which) {
                case 1: r = s.algo.getWMGMLevelset(); break;
                case 2: r = s.algo.getGMCSFLevelset(); break;
                case 3: r = s.algo.getCentralLevelset(); break;
                case 4: r = s.algo.getCorticalThickness(); break;
                case 5: r = s.algo.getCerebralWMprobability(); break;
                case 6: r = s.algo.getCorticalGMprobability(); break;
                default: r = s.algo.getSulcalCSFprobability(); break;
            }
            if (r.length != outLen) { s.lastError = "size mismatch"; return -2; }
            for (int i = 0; i < outLen; i++) outPtr.write(i, r[i]);
            return 0;
        } catch (Throwable t) { s.lastError = NativeArrayUtil.describe(t); return -1; }
    }

    @CEntryPoint(name = "crashs_cruise_get_last_error")
    public static void getLastError(IsolateThread thread, ObjectHandle handle, CCharPointer buf, int bufLen) {
        State s = ObjectHandles.getGlobal().get(handle);
        CTypeConversion.toCString(s.lastError == null ? "" : s.lastError, buf, org.graalvm.word.WordFactory.unsigned(bufLen));
    }

    @CEntryPoint(name = "crashs_cruise_destroy")
    public static void destroy(IsolateThread thread, ObjectHandle handle) {
        ObjectHandles.getGlobal().destroy(handle);
    }
}
