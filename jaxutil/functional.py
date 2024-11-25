from jaxutil.tree import tree_map
from jax.lax import scan
import numpy as np

# unvectorized map - functionally equivalent to vmap, but does not perform a vectorizing transformation.
# sometimes the vmap transformation can result in poor performance when the mapped function contains control
# flow that affects the computational cost. in these cases it can be beneficial to use umap instead, which 
# essentially just wraps scan with a vmap-like interface.
def umap(f, in_axes=None):
    def _umap(*inputs):
        mapped_inputs = []
        unmapped_inputs = []
        is_mapped = lambda _: in_axes is None or in_axes[_] is not None
        for i in range(len(inputs)):
            if is_mapped(i):
                ax = 0 if in_axes is None else in_axes[i]
                mapped_inputs.append(
                    tree_map(lambda _: _.transpose(ax, *list(range(ax)), *list(range(ax+1, _.ndim))), inputs[i]))
            else:
                unmapped_inputs.append(inputs[i])
        def _f(carry, mapped_inputs):
            x = []
            mapped_idx = 0
            unmapped_idx = 0
            for i in range(len(inputs)):
                if is_mapped(i):
                    x.append(mapped_inputs[mapped_idx])
                    mapped_idx += 1
                else:
                    x.append(unmapped_inputs[unmapped_idx])
                    unmapped_idx += 1
            return None, f(*x)
        return scan(_f, None, mapped_inputs)[1]
    return _umap