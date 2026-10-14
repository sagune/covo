import unittest
import numpy as np
from padding_fix import apply_mask_to_padding_states


class Tensor:
    """Small NumPy-backed test double for the exact tensor operations used by the patch."""
    def __init__(self, value):self.value=np.asarray(value)
    @property
    def dtype(self):return self.value.dtype
    @property
    def shape(self):return self.value.shape
    def __getitem__(self, key):return Tensor(self.value[key])
    def __mul__(self, other):return Tensor(self.value*other.value)
    def to(self, dtype):return Tensor(self.value.astype(dtype))


class Tests(unittest.TestCase):
    def test_single_padding_zeroed(self):
        x=Tensor(np.arange(12,dtype=np.float32).reshape(1,3,4)+1)
        y=apply_mask_to_padding_states(x,Tensor([[0,1,1]]))
        np.testing.assert_array_equal(y.value[:,0],0)
        np.testing.assert_array_equal(y.value[:,1:],x.value[:,1:])
        self.assertEqual(y.dtype,x.dtype)

    def test_batch_unchanged_semantics(self):
        x=Tensor(np.ones((2,3,4),dtype=np.float32));mask=Tensor([[0,1,1],[1,1,1]])
        y=apply_mask_to_padding_states(x,mask)
        np.testing.assert_array_equal(y.value,x.value*mask.value[:,:,None])

    def test_none_and_all_one(self):
        x=Tensor(np.ones((1,3,4),dtype=np.float16))
        self.assertIs(apply_mask_to_padding_states(x,None),x)
        np.testing.assert_array_equal(apply_mask_to_padding_states(x,Tensor([[1,1,1]])).value,x.value)

    def test_cached_single_step_guard_preserved(self):
        x=Tensor(np.ones((1,1,4),dtype=np.float32))
        self.assertIs(apply_mask_to_padding_states(x,Tensor([[0]])),x)


if __name__=='__main__':unittest.main()
