"""
This script is licensed under the MIT License.
For more details, see the LICENSE file in the root directory of this repository.

(c) 2024 Lukas Thede
"""

"""
Random Solarization
"""
from numpy.random import rand
from typing import Callable, Optional, Tuple
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
from ffcv.pipeline.compiler import Compiler

class RandomSolarization(Operation):
    """Solarize the image randomly with a given probability by inverting all pixel
    values above a threshold. If img is a Tensor, it is expected to be in [..., 1 or 3, H, W] format,
    where ... means it can have an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".
    Parameters
    ----------
        p (float): probability of the image being solarized. Default value is 0.5
        threshold (float): all pixels equal or above this value are inverted.
    """

    def __init__(self, p: float = 0.5, threshold: float = 128):
        super().__init__()
        self.p = p
        self.threshold = threshold

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        threshold = self.threshold
        p = self.p

        def solarize(images, dst):
            should_solarize = rand(images.shape[0]) < p
            for i in my_range(images.shape[0]):
                if should_solarize[i]:
                    mask = (images[i] >= threshold)
                    dst[i] = images[i] * (1-mask) + (255 - images[i])*mask
                else:
                    dst[i] = images[i]
            return dst

        solarize.is_parallel = True
        return solarize

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        assert previous_state.jit_mode
        return (previous_state, AllocationQuery(previous_state.shape, previous_state.dtype))
