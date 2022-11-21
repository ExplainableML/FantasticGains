import ffcv

class RandomHorizontalFlip(ffcv.transforms.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super().__init__(flip_prob=p)
