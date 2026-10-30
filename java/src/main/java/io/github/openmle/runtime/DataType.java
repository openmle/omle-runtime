package io.github.openmle.runtime;

/** Element data type — mirrors omle_dtype_t. */
public enum DataType {
    UNKNOWN(0),
    BOOL(1),
    INT8(2),
    INT16(3),
    INT32(4),
    INT64(5),
    UINT8(6),
    UINT16(7),
    UINT32(8),
    UINT64(9),
    FLOAT16(10),
    FLOAT32(11),
    FLOAT64(12),
    STRING(13),
    BYTES(14),
    DATE(15),
    TIME(16),
    TIMESTAMP(17);

    private final int code;

    DataType(int code) { this.code = code; }

    public int code() { return code; }

    public static DataType fromCode(int code) {
        for (DataType dt : values())
            if (dt.code == code) return dt;
        return UNKNOWN;
    }
}
