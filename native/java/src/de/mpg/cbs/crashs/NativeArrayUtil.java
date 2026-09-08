package de.mpg.cbs.crashs;

import org.graalvm.nativeimage.c.type.CFloatPointer;
import org.graalvm.nativeimage.c.type.CIntPointer;

final class NativeArrayUtil {
    private NativeArrayUtil() {}

    static float[] toFloatArray(CFloatPointer ptr, int n) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) a[i] = ptr.read(i);
        return a;
    }

    static int[] toIntArray(CIntPointer ptr, int n) {
        int[] a = new int[n];
        for (int i = 0; i < n; i++) a[i] = ptr.read(i);
        return a;
    }

    static void copyOut(float[] src, CFloatPointer outPtr, int outLen) {
        for (int i = 0; i < outLen; i++) outPtr.write(i, src[i]);
    }

    static void copyOut(int[] src, CIntPointer outPtr, int outLen) {
        for (int i = 0; i < outLen; i++) outPtr.write(i, src[i]);
    }

    static String describe(Throwable t) {
        java.io.StringWriter sw = new java.io.StringWriter();
        t.printStackTrace(new java.io.PrintWriter(sw));
        return sw.toString();
    }
}
