package io.github.openmle.runtime;

import java.util.Arrays;

/** Describes one model input slot. */
public final class InputSpec {

    private final String   name;
    private final DataType dtype;
    private final long[]   shape;

    InputSpec(String name, DataType dtype, long[] shape) {
        this.name  = name;
        this.dtype = dtype;
        this.shape = shape.clone();
    }

    public String   name()  { return name; }
    public DataType dtype() { return dtype; }
    public long[]   shape() { return shape.clone(); }

    @Override
    public String toString() {
        return "InputSpec{name='" + name + "', dtype=" + dtype + ", shape=" + Arrays.toString(shape) + '}';
    }
}
