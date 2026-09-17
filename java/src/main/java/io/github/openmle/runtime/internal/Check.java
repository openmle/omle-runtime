package io.github.openmle.runtime.internal;

import io.github.openmle.runtime.OMLEException;

/** Helper to convert a C status code into a Java exception. */
public final class Check {

    private Check() {}

    public static void ok(int status) {
        if (status != 0) {
            String msg = LibHolder.LIB.omle_last_error();
            throw new OMLEException(status, msg != null ? msg : "omle error " + status);
        }
    }
}
