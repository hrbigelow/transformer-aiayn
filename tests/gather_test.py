import jax
import jax.numpy as jnp


"""
offset_dims – the set of dimensions in the gather output that offset into an array
sliced from operand. Must be a tuple of integers in ascending order, each
representing a dimension number of the output.
"""
offset_dims = (0,)

"""
collapsed_slice_dims – the set of dimensions i in operand that have slice_sizes[i] ==
1 and that should not have a corresponding dimension in the output of the gather.
Must be a tuple of integers in ascending order.

Sometimes I get this error:

TypeError: All components of the offset index in a gather op must either be a 
offset dimension or explicitly collapsed; got len(slice_sizes)=2, output_slice_sizes=(0, 1),
collapsed_slice_dims=(0,). 

I believe the phrase 'offset index' here refers to the 'start_indices' argument.

"""
collapsed_slice_dims = (0,)

"""
start_index_map – for each dimension in start_indices, gives the corresponding
dimension in operand that is to be sliced. 

*** This is the key ***
Must be a tuple of integers with size equal to start_indices.shape[-1].  
"""
start_index_map = (0,)

gd = jax.lax.GatherDimensionNumbers(offset_dims, collapsed_slice_dims, start_index_map)


"""
the size of each slice. Must be a sequence of non-negative integers with length equal
to ndim(operand).  

This makes sense.  In this terminology, a 'slice' is really a hyper-rectangular
region of the operand, with the same rank as the operand.  This region is defined by
an offset and a size.  Both the offset and size also are tuples of 
"""
slice_sizes = (1,20)

operand = jnp.arange(5*20).reshape(5,20)

# start_indices = jnp.array([[1,0], [2,0], [4,0]])
"""
"""
start_indices = jnp.array([2,0,0,1,2,4]).reshape(3,2,1)
# start_indices = jnp.array([[[2],[0],[0]], [[1],[2],[4]]])

output = jax.lax.gather(operand, start_indices, gd, slice_sizes)

print(f'{output.shape} = gather({operand.shape}, {start_indices.shape}, '
        f'GD({offset_dims}, {collapsed_slice_dims}, {start_index_map})',
        f'{slice_sizes})')
# print(output)

