package io.github.openmle.runtime;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for per-thread Session execution.
 */
class SessionTest extends TestBase {

    private static byte[] MODEL_BYTES;

    @BeforeAll
    static void loadFixture() {
        MODEL_BYTES = modelBytes2f();
    }

    // ---------------------------------------------------------------------------
    // Basic creation
    // ---------------------------------------------------------------------------

    @Test
    void createAndClose() {
        try (Model m = Model.loadBytes(MODEL_BYTES);
             Session s = m.createSession()) {
            assertNotNull(s);
        }
    }

    @Test
    void toStringContainsOutputs() {
        try (Model m = Model.loadBytes(MODEL_BYTES);
             Session s = m.createSession()) {
            assertTrue(s.toString().contains("Session"));
        }
    }

    @Test
    void numOutputs() {
        try (Model m = Model.loadBytes(MODEL_BYTES);
             Session s = m.createSession()) {
            assertEquals(1, s.numOutputs());
        }
    }

    // ---------------------------------------------------------------------------
    // Numerical correctness
    // ---------------------------------------------------------------------------

    @Test
    void predictLeftBranch() {
        try (Model m = Model.loadBytes(MODEL_BYTES);
             Session s = m.createSession()) {
            float[] out = s.predict(new float[][]{{0.0f, 0f}});
            assertClose(2.0f, out[0], 1e-5f);
        }
    }

    @Test
    void predictRightBranch() {
        try (Model m = Model.loadBytes(MODEL_BYTES);
             Session s = m.createSession()) {
            float[] out = s.predict(new float[][]{{1.0f, 0f}});
            assertClose(-2.0f, out[0], 1e-5f);
        }
    }

    @Test
    void predictBoundary() {
        try (Model m = Model.loadBytes(MODEL_BYTES);
             Session s = m.createSession()) {
            float[] out = s.predict(new float[][]{{0.5f, 0f}});
            assertClose(-2.0f, out[0], 1e-5f);
        }
    }

    // ---------------------------------------------------------------------------
    // predict overloads
    // ---------------------------------------------------------------------------

    @Test
    void predictFlatArray() {
        try (Model m = Model.loadBytes(MODEL_BYTES);
             Session s = m.createSession()) {
            float[] X   = {0.1f, 0f, 0.9f, 0f};
            float[] out = s.predict(X, 2, 2);
            assertClose(2.0f,  out[0], 1e-5f);
            assertClose(-2.0f, out[1], 1e-5f);
        }
    }

    @Test
    void predictSingleRow() {
        try (Model m = Model.loadBytes(MODEL_BYTES);
             Session s = m.createSession()) {
            float[] out = s.predict(new float[][]{{0.1f, 0f}});
            assertEquals(1, out.length);
            assertClose(2.0f, out[0], 1e-5f);
        }
    }

    // ---------------------------------------------------------------------------
    // run() → Tensor
    // ---------------------------------------------------------------------------

    @Test
    void runReturnsTensor() {
        try (Model m = Model.loadBytes(MODEL_BYTES);
             Session s = m.createSession()) {
            Tensor t = s.run(new float[][]{{0.1f, 0f}});
            assertNotNull(t);
            assertEquals(1, t.nRows());
        }
    }

    @Test
    void runTensorValue() {
        try (Model m = Model.loadBytes(MODEL_BYTES);
             Session s = m.createSession()) {
            Tensor t = s.run(new float[][]{{0.1f, 0f}});
            assertClose(2.0f, (float) t.get(0, 0), 1e-5f);
        }
    }

    @Test
    void runBorrowedTensorShouldNotBeClosed() {
        try (Model m = Model.loadBytes(MODEL_BYTES);
             Session s = m.createSession()) {
            Tensor t = s.run(new float[][]{{0.1f, 0f}});
            // Borrowed Tensor — calling close() should be a no-op (owned=false).
            assertDoesNotThrow(t::close);
            // Still readable after no-op close.
            assertClose(2.0f, (float) t.get(0, 0), 1e-5f);
        }
    }

    // ---------------------------------------------------------------------------
    // Reuse across calls
    // ---------------------------------------------------------------------------

    @Test
    void reuseAcrossCalls() {
        try (Model m = Model.loadBytes(MODEL_BYTES);
             Session s = m.createSession()) {
            float[][] cases = {{0.1f, 0f}, {0.9f, 0f}, {0.2f, 0f}, {0.8f, 0f}};
            float[]   expect = {2.0f, -2.0f, 2.0f, -2.0f};
            for (int i = 0; i < cases.length; i++) {
                float[] out = s.predict(new float[][]{cases[i]});
                assertClose(expect[i], out[0], 1e-5f);
            }
        }
    }

    @Test
    void sessionMatchesModelPredict() {
        try (Model m = Model.loadBytes(MODEL_BYTES);
             Session s = m.createSession()) {
            Random rng = new Random(7);
            float[][] X = new float[50][2];
            for (int r = 0; r < 50; r++) {
                X[r][0] = (float) rng.nextGaussian();
                X[r][1] = (float) rng.nextGaussian();
            }
            float[] fromModel   = m.predict(X);
            float[] fromSession = s.predict(X);
            assertArrayEquals(fromModel, fromSession, 1e-5f);
        }
    }

    // ---------------------------------------------------------------------------
    // Independent sessions
    // ---------------------------------------------------------------------------

    @Test
    void twoSessionsGiveSameResult() {
        try (Model m = Model.loadBytes(MODEL_BYTES);
             Session s1 = m.createSession();
             Session s2 = m.createSession()) {
            float[][] X = {{0.1f, 0f}, {0.9f, 0f}};
            assertArrayEquals(s1.predict(X), s2.predict(X), 1e-5f);
        }
    }

    @Test
    void batchCorrectness() {
        try (Model m = Model.loadBytes(MODEL_BYTES);
             Session s = m.createSession()) {
            Random rng = new Random(99);
            int n = 200;
            float[][] X = new float[n][2];
            for (int i = 0; i < n; i++) {
                X[i][0] = (float) rng.nextGaussian();
                X[i][1] = (float) rng.nextGaussian();
            }
            float[] out = s.predict(X);
            for (int i = 0; i < n; i++) {
                float expected = X[i][0] < 0.5f ? 2.0f : -2.0f;
                assertClose(expected, out[i], 1e-5f);
            }
        }
    }
}
