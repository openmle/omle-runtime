package io.github.openmle.runtime;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for Model loading, schema, and stateless predict.
 * Fixture: 2-feature stump, feat0 < 0.5 → +2.0, else −2.0.
 */
class ModelTest extends TestBase {

    private static byte[] MODEL_BYTES;

    @BeforeAll
    static void loadFixture() {
        MODEL_BYTES = modelBytes2f();
    }

    // ---------------------------------------------------------------------------
    // Loading
    // ---------------------------------------------------------------------------

    @Test
    void loadBytesReturnsModel() {
        try (Model m = Model.loadBytes(MODEL_BYTES)) {
            assertNotNull(m);
        }
    }

    @Test
    void loadFileReturnsModel() throws Exception {
        java.nio.file.Path tmp = java.nio.file.Files.createTempFile("model_", ".omle");
        java.nio.file.Files.write(tmp, MODEL_BYTES);
        try (Model m = Model.loadFile(tmp.toString())) {
            assertNotNull(m);
        } finally {
            java.nio.file.Files.deleteIfExists(tmp);
        }
    }

    @Test
    void badBytesThrows() {
        assertThrows(OMLEException.class, () -> Model.loadBytes(new byte[]{0x01, 0x02, 0x03}));
    }

    @Test
    void missingFileThrows() {
        assertThrows(OMLEException.class, () -> Model.loadFile("/no/such/file.omle"));
    }

    @Test
    void multipleLoadsAreIndependent() {
        try (Model m1 = Model.loadBytes(MODEL_BYTES);
             Model m2 = Model.loadBytes(MODEL_BYTES)) {
            assertNotSame(m1, m2);
            assertEquals(m1.numInputs(), m2.numInputs());
        }
    }

    // ---------------------------------------------------------------------------
    // Schema
    // ---------------------------------------------------------------------------

    @Test
    void numInputs() {
        try (Model m = Model.loadBytes(MODEL_BYTES)) {
            assertEquals(1, m.numInputs());
        }
    }

    @Test
    void numOutputs() {
        try (Model m = Model.loadBytes(MODEL_BYTES)) {
            assertEquals(1, m.numOutputs());
        }
    }

    @Test
    void inputSpecNames() {
        try (Model m = Model.loadBytes(MODEL_BYTES)) {
            assertEquals("X", m.inputSpec(0).name());
        }
    }

    @Test
    void outputSpecName() {
        try (Model m = Model.loadBytes(MODEL_BYTES)) {
            assertEquals("score", m.outputSpec(0).name());
        }
    }

    @Test
    void inputSpecDtype() {
        try (Model m = Model.loadBytes(MODEL_BYTES)) {
            for (InputSpec s : m.inputs())
                assertNotNull(s.dtype());
        }
    }

    @Test
    void toStringContainsShape() {
        try (Model m = Model.loadBytes(MODEL_BYTES)) {
            String s = m.toString();
            assertTrue(s.contains("1"), "numInputs not in toString");
        }
    }

    // ---------------------------------------------------------------------------
    // Numerical correctness
    // ---------------------------------------------------------------------------

    @Test
    void leftBranch() {
        // feat0 = 0.0 < 0.5  →  score = +2.0
        try (Model m = Model.loadBytes(MODEL_BYTES)) {
            float[] out = m.predict(new float[][]{{0.0f, 99f}});
            assertClose(2.0f, out[0], 1e-5f);
        }
    }

    @Test
    void rightBranch() {
        // feat0 = 1.0 >= 0.5  →  score = −2.0
        try (Model m = Model.loadBytes(MODEL_BYTES)) {
            float[] out = m.predict(new float[][]{{1.0f, 99f}});
            assertClose(-2.0f, out[0], 1e-5f);
        }
    }

    @Test
    void boundaryIsStrictLessThan() {
        // feat0 = 0.5 is NOT < 0.5  →  score = −2.0
        try (Model m = Model.loadBytes(MODEL_BYTES)) {
            float[] out = m.predict(new float[][]{{0.5f, 0f}});
            assertClose(-2.0f, out[0], 1e-5f);
        }
    }

    @Test
    void feat1IsIgnored() {
        try (Model m = Model.loadBytes(MODEL_BYTES)) {
            for (float f1 : new float[]{-100f, 0f, 100f}) {
                float[] out = m.predict(new float[][]{{0.1f, f1}});
                assertClose(2.0f, out[0], 1e-5f);
            }
        }
    }

    @Test
    void batchCorrectness() {
        try (Model m = Model.loadBytes(MODEL_BYTES)) {
            Random rng = new Random(42);
            int n = 500;
            float[][] X = new float[n][2];
            for (int i = 0; i < n; i++) {
                X[i][0] = (float) rng.nextGaussian();
                X[i][1] = (float) rng.nextGaussian();
            }
            float[] out = m.predict(X);
            assertEquals(n, out.length);
            for (int i = 0; i < n; i++) {
                float expected = X[i][0] < 0.5f ? 2.0f : -2.0f;
                assertClose(expected, out[i], 1e-5f);
            }
        }
    }

    // ---------------------------------------------------------------------------
    // predict overloads
    // ---------------------------------------------------------------------------

    @Test
    void predictFlatArray() {
        try (Model m = Model.loadBytes(MODEL_BYTES)) {
            // Two rows: [0.1, 0.0] and [0.9, 0.0]
            float[] X = {0.1f, 0.0f, 0.9f, 0.0f};
            float[] out = m.predict(X, 2, 2);
            assertClose(2.0f,  out[0], 1e-5f);
            assertClose(-2.0f, out[1], 1e-5f);
        }
    }

    @Test
    void predictSingleRow() {
        try (Model m = Model.loadBytes(MODEL_BYTES)) {
            float[] out = m.predict(new float[][]{{0.1f, 0f}});
            assertEquals(1, out.length);
            assertClose(2.0f, out[0], 1e-5f);
        }
    }

    // ---------------------------------------------------------------------------
    // Session factory
    // ---------------------------------------------------------------------------

    @Test
    void createSessionReturnsSession() {
        try (Model m = Model.loadBytes(MODEL_BYTES);
             Session s = m.createSession()) {
            assertNotNull(s);
        }
    }
}
