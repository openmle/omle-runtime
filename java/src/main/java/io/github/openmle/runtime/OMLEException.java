package io.github.openmle.runtime;

/** Thrown when the omle C library returns a non-zero status code. */
public class OMLEException extends RuntimeException {

    private final int statusCode;

    public OMLEException(int statusCode, String message) {
        super(message);
        this.statusCode = statusCode;
    }

    /** Raw omle_status_t value (0 = OK). */
    public int getStatusCode() {
        return statusCode;
    }
}
