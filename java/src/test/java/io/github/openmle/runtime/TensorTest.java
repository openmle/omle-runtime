package io.github.openmle.runtime;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class TensorTest extends TestBase {

    @Test
    void createFromArray() {
        float[][] data = {{1f, 2f}, {3f, 4f}};
        try (Tensor t = Tensor.of(data)) {
            assertEquals(2, t.nRows());
            assertEquals(2, t.nCols());
            assertEquals(4, t.numel());
        }
    }

    @Test
    void roundtripToArray() {
        float[][] data = {{1.5f, 2.5f}, {3.5f, 4.5f}};
        try (Tensor t = Tensor.of(data)) {
            float[][] result = t.toArray();
            assertArrayEquals(data[0], result[0], 1e-6f);
            assertArrayEquals(data[1], result[1], 1e-6f);
        }
    }

    @Test
    void roundtripFlatArray() {
        float[] flat = {1f, 2f, 3f, 4f, 5f, 6f};
        try (Tensor t = Tensor.ofFlat(flat, 2, 3)) {
            assertEquals(2, t.nRows());
            assertEquals(3, t.nCols());
            assertArrayEquals(flat, t.toFlatArray(), 1e-6f);
        }
    }

    @Test
    void getElement() {
        float[][] data = {{10f, 20f}, {30f, 40f}};
        try (Tensor t = Tensor.of(data)) {
            assertEquals(10.0, t.get(0, 0), 1e-6);
            assertEquals(20.0, t.get(0, 1), 1e-6);
            assertEquals(30.0, t.get(1, 0), 1e-6);
            assertEquals(40.0, t.get(1, 1), 1e-6);
        }
    }

    @Test
    void toStringContainsShape() {
        try (Tensor t = Tensor.of(new float[][]{{1f, 2f}})) {
            String s = t.toString();
            assertTrue(s.contains("1"), "rows not in toString");
            assertTrue(s.contains("2"), "cols not in toString");
        }
    }

    @Test
    void jagged2dThrows() {
        assertThrows(IllegalArgumentException.class, () -> {
            float[][] data = {{1f, 2f}, {3f}};
            try (Tensor t = Tensor.of(data)) {
                t.toFlatArray();  // force evaluation
            }
        });
    }

    @Test
    void emptyDataThrows() {
        assertThrows(IllegalArgumentException.class, () -> Tensor.of(new float[0][]));
    }

    @Test
    void closedTensorThrows() {
        Tensor t = Tensor.of(new float[][]{{1f}});
        t.close();
        assertThrows(IllegalStateException.class, t::toFlatArray);
    }
}
