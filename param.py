

class Param:

    def __init__(self):
        #preprocessing
        self.path = '/home/filip/Dokumenty/FAV/DIP/github/MNE_ML/PROJECT_DAYS_P3_NUMBERS/'
        self.l_freq = 0.1
        self.h_freq = 30
        self.t_min = -0.2
        self.t_max = 1
        self.baseline = (-0.2, 0)
        self.amplitude = 150e-6
        self.scaling = 1e6

        #features extraction
        self.min_latency = 0.3
        self.max_latency = 1
        self.steps = 20

        #classification parameters
        self.test_part = 0.25
        self.validation_part = 0.25
        self.cross_val_iter = 30
        self.epochs = 30
        self.verbose = 2
