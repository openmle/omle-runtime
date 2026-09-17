package io.github.openmle.runtime;

import java.util.Arrays;

/** Describes one model output slot. */
public final class OutputSpec {

    private final String   name;
    private final DataType dtype;
    private final long[]   shape;

    OutputSpec(String name, DataType dtype, long[] shape) {
        this.name  = name;
        this.dtype = dtype;
        this.shape = shape.clone();
    }

    public String   name()  { return name; }
    public DataType dtype() { return dtype; }
    public long[]   shape() { return shape.clone(); }

    @Override
    public String toString() {
        return "OutputSpec{name='" + name + "', dtype=" + dtype + ", shape=" + Arrays.toString(shape) + '}';
    }
}
