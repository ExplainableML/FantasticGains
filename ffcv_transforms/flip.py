"""
This script is licensed under the MIT License.
For more details, see the LICENSE file in the root directory of this repository.

(c) 2024 Lukas Thede
"""

import ffcv

class RandomHorizontalFlip(ffcv.transforms.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super().__init__(flip_prob=p)
