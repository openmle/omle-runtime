package io.github.openmle.runtime.internal;

import com.sun.jna.Native;

/** Lazy singleton — loaded once, shared by all public API classes. */
public final class LibHolder {

    public static final Lib LIB = load();

    private LibHolder() {}

    private static Lib load() {
        // Allow callers to add an extra search directory at startup via
        // -Dio.github.openmle.libpath=<dir>  (supplements jna.library.path).
        String extra = System.getProperty("io.github.openmle.libpath");
        if (extra != null && !extra.isEmpty()) {
            String existing = System.getProperty("jna.library.path", "");
            System.setProperty("jna.library.path",
                existing.isEmpty() ? extra : existing + java.io.File.pathSeparator + extra);
        }
        return Native.load("omleruntime", Lib.class);
    }
}
