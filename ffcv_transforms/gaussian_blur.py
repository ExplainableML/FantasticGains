"""
This script is licensed under the MIT License.
For more details, see the LICENSE file in the root directory of this repository.

(c) 2024 Lukas Thede
"""

"""
Random Solarization
"""
from typing import Callable, Optional, Tuple

from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
from ffcv.pipeline.compiler import Compiler
import numpy as np
from numpy.random import rand
import numba as nb
from scipy.signal import convolve2d
import copy

class RandomGaussianBlur(Operation):
    """Blurs image with randomly chosen Gaussian blur.
    Parameters
    ----------
        sigma ((float, float)): random sample standard deviation from [min = sigma[0], max = sigma[1]].
        truncate (float): scaling value for kernel size based on sigma: kernel_size = int(truncate * sigma + 0.5).
        p (float): probability of blurring the image.
    """

    def __init__(
        self,
        sigma: Tuple[float] = (0.3, 2.0),
        truncate: float = 4.0,
        p: float = 0.5
    ):
        super().__init__()
        self.truncate = truncate
        self.sigma = sigma
        self.p = p

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        p = self.p
        sigma = self.sigma
        truncate = self.truncate

        def gaussian_blur(images, dst):
            should_blur = rand(images.shape[0]) < p
            random_sigmas = np.random.uniform(sigma[0], sigma[1], images.shape[0])
            kernel_sizes = (truncate * random_sigmas + 0.5).astype(np.int32)

            for i in my_range(images.shape[0]):
                kernel_size = kernel_sizes[i]
                if should_blur[i] and kernel_size > 1:
                    img = images[i]
                    ksize_half_x = (kernel_size - 1) * 0.5
                    ksize_half_y = (kernel_size - 1) * 0.5
                    x = np.linspace(-ksize_half_x, ksize_half_x, kernel_size)
                    y = np.linspace(-ksize_half_y, ksize_half_y, kernel_size)
                    pdf_x = np.exp(-0.5 * (x / random_sigmas[i]) ** 2)
                    pdf_y = np.exp(-0.5 * (y / random_sigmas[i]) ** 2)
                    kernel1d_x = pdf_x / pdf_x.sum()
                    kernel1d_y = pdf_y / pdf_y.sum()
                    kernel_2d = np.dot(np.expand_dims(kernel1d_y, axis=-1), np.expand_dims(kernel1d_x, axis=0))
                    padding = [[kernel_size // 2, kernel_size // 2], [kernel_size // 2, kernel_size // 2]]

                    output = np.zeros_like(img)
                    for k in range(img.shape[-1]):
                        # pad input image.
                        new_shape = (
                            img.shape[0] + padding[0][0] + padding[0][1],
                            img.shape[1] + padding[1][0] + padding[1][1]
                        )
                        pad_img = np.empty(new_shape, dtype=img[k].dtype)

                        original_area_slice = (
                            slice(padding[0][0], padding[0][0] + img.shape[0]),
                            slice(padding[1][0], padding[1][0] + img.shape[1])
                        )

                        pad_img[original_area_slice] = img[:, :, k]

                        axes = range(pad_img.ndim)

                        # for m in range(2):
                        for axis, (left_index, right_index) in zip(axes, padding):
                            count = 0
                            while left_index > 0 or right_index > 0:
                                count = count + 1
                                edge_offset = 0  # Edge is not included, no need to offset pad amount
                                old_length = pad_img.shape[axis] - right_index - left_index - count  # but must be omitted from the chunk
                                if left_index > 0:
                                    old_longer = old_length > left_index
                                    # Pad with reflected values on left side:
                                    # First limit chunk size which can't be larger than pad area
                                    # if old_length > left_index:
                                    #     chunk_length = left_index
                                    # else:
                                    #     chunk_length = old_length
                                    # Slice right to left, stop on or next to edge, start relative to stop
                                    base_stop = left_index - edge_offset
                                    if old_longer:
                                        base_start = base_stop + left_index
                                    else:
                                        base_start = base_stop + old_length
                                    # if axis == 0:
                                    #     left_chunk = pad_img[slice(base_start, base_stop, -1), original_area_slice[1]]
                                    # if axis == 1:
                                    #     left_chunk = pad_img[:, slice(base_start, base_stop, -1)]
                                    # Insert chunk into roi area
                                    if old_longer:
                                        start = 0
                                    else:
                                        start = left_index - old_length
                                    stop = left_index
                                    if axis == 0:
                                        pad_img[slice(start, stop), original_area_slice[1]] = pad_img[slice(base_start, base_stop, -1), original_area_slice[1]]
                                    if axis == 1:
                                        pad_img[:, slice(start, stop)] = pad_img[:, slice(base_start, base_stop, -1)]
                                    # if axis == 0:
                                    #     pad_img[slice(start, stop), original_area_slice[1]] = left_chunk
                                    # if axis == 1:
                                    #     pad_img[:, slice(start, stop)] = left_chunk
                                    # Adjust pointer to left edge for next iteration
                                    if old_longer:
                                        left_index = 0
                                    else:
                                        left_index = left_index - old_length

                                if right_index > 0:
                                    # Pad with reflected values on right side:
                                    # First limit chunk size which can't be larger than pad area
                                    old_longer = old_length > right_index
                                    # Slice right to left, start on or next to edge, stop relative to start
                                    base_start = -right_index + edge_offset - 2
                                    if old_longer:
                                        base_stop = base_start - right_index
                                    else:
                                        base_stop = base_start - old_length
                                    if old_longer:
                                        if axis == 0:
                                            pad_img[slice(pad_img.shape[0] - right_index, pad_img.shape[0] - right_index + old_length), original_area_slice[1]] = pad_img[slice(base_start, base_stop, -1), original_area_slice[1]]
                                        if axis == 1:
                                            pad_img[:, slice(pad_img.shape[1] - right_index, pad_img.shape[1] - right_index + old_length)] = pad_img[:, slice(base_start, base_stop, -1)]
                                    else:
                                        if axis == 0:
                                            pad_img[slice(pad_img.shape[0] - right_index, pad_img.shape[0]), original_area_slice[1]] = pad_img[slice(base_start, base_stop, -1), original_area_slice[1]]
                                        if axis == 1:
                                            pad_img[:, slice(pad_img.shape[1] - right_index, pad_img.shape[1])] = pad_img[:, slice(base_start, base_stop, -1)]
                                    # Adjust pointer to right edge for next iteration
                                    if old_longer:
                                        right_index = 0
                                    else:
                                        right_index = right_index - old_length

                        # Run convolution with kernel.
                        n_rows, n_cols = pad_img.shape
                        shift = kernel_size % 2
                        for rr in range(n_rows - kernel_size + shift):
                            for cc in range(n_cols - kernel_size + shift):
                                output[rr, cc, k] = np.sum(pad_img[rr:rr+kernel_size, cc:cc+kernel_size] * kernel_2d)
                    dst[i] = output
                else:
                    dst[i] = images[i]
            return dst

        gaussian_blur.is_parallel = True
        return gaussian_blur

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        assert previous_state.jit_mode
        return (previous_state, AllocationQuery(previous_state.shape, previous_state.dtype))
