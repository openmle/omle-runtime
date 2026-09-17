package io.github.openmle.runtime;

import java.io.InputStream;

/** Shared test utilities. */
class TestBase {

    /** Load the 2-feature stump model bytes from test resources. */
    static byte[] modelBytes2f() {
        try (InputStream in = TestBase.class.getResourceAsStream("/test_model_2f.omle")) {
            if (in == null)
                throw new IllegalStateException("test_model_2f.omle not found in test resources");
            return in.readAllBytes();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Assert two floats are within relative tolerance.
     *
     * @param expected reference value
     * @param actual   computed value
     * @param rtol     relative tolerance (e.g. 1e-5f)
     */
    static void assertClose(float expected, float actual, float rtol) {
        float tol = rtol * Math.abs(expected) + 1e-8f;
        if (Math.abs(actual - expected) > tol)
            throw new AssertionError(
                "expected " + expected + " but got " + actual
                + " (diff=" + Math.abs(actual - expected) + ", tol=" + tol + ")");
    }
}
