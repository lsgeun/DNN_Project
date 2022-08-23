from FMD import FMD

class FMDs(FMD):
    fmd1 = None; fmd2 = None
    def __init__(self):
        self.fmd1 = FMD("/Users/macbookair_sg/Library/Mobile Documents/com~apple~CloudDocs/대학/졸업/졸업 과제/data_sets/cifar/cifar10_2/cat")
        self.fmd2 = FMD("/Users/macbookair_sg/Library/Mobile Documents/com~apple~CloudDocs/대학/졸업/졸업 과제/data_sets/cifar/cifar10_2/airplane")