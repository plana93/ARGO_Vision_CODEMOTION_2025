import numpy as np

def print_shape(**kwargs):
    """
    Print detailed shape, type, and preview for each input variable.
    Handles ragged nested lists and numpy arrays gracefully.
    """

    def _safe_to_np(x):
        try:
            return np.array(x, dtype=object if isinstance(x, (list, tuple)) else None)
        except Exception:
            return np.array([None], dtype=object)

    def _describe_shape(x):
        """Return detailed shape info, including sub-lengths if ragged."""
        try:
            arr = np.array(x, dtype=object)
            if arr.dtype == object:
                # ragged: collect sub-shapes
                sub_shapes = []
                for a in arr:
                    try:
                        sub_shapes.append(np.array(a).shape)
                    except Exception:
                        sub_shapes.append("N/A")
                return f"ragged [{len(arr)}] (sub-shapes: {sub_shapes})"
            else:
                return str(arr.shape)
        except Exception:
            return "N/A"

    def _short_preview(x, maxlen=100):
        s = str(x).replace("\n", " ")
        return s[:maxlen] + ("..." if len(s) > maxlen else "")

    print("\n🔍 === DEBUG: Input Data Overview ===")
    for name, value in kwargs.items():
        vtype = type(value).__name__
        shape_info = _describe_shape(value)
        preview = _short_preview(value)

        print(f"\n• {name:<12}: type={vtype:<10} | shape={shape_info}")
        print(f"  content     : {preview}")
    print("=====================================\n")
