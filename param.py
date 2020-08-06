class Param:
    """
    Configuration class
    """

    def __init__(self):
        """
        Initializes the configuration class
        """
        # preprocessing
        # path to the data directory in the absolute form
        self.path = '/home/filip/Dokumenty/FAV/DIP/github/MNE_ML/PROJECT_DAYS_P3_NUMBERS/'
        self.l_freq = 0.1  # low-pass filter
        self.h_freq = 30  # high-pass filter
        self.t_min = -0.2  # first time value of the interval for the epoch extraction
        self.t_max = 1  # second time value of the interval for the epoch extraction
        self.baseline = (-0.2, 0)  # baseline correction interval
        self.amplitude = 150e-6  # peak-to-peak amplitude for the artifact removal
        self.scaling = 1e6  # conversion from μV to V

        # features extraction
        self.min_latency = 0.3  # first time value of the interval for the features extraction
        self.max_latency = 1  # second time value of the interval for the features extraction
        self.steps = 20  # number of features

        # classification parameters
        self.test_part = 0.25  # part of the data for the test set
        self.validation_part = 0.25  # part of the data for the validation set
        self.cross_val_iter = 30  # number of the iterations of the Monte*carlo cross-validation
        self.epochs = 30  # number of the iterations of the neural network training
        self.verbose = 2  # verbose parameter
