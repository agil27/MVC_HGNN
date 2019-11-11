class config(object):
    def __init__(self):
        super(config, self).__init__()

        # default
        self.data_root = 'data'
        self.data_deap = 'data'
        self.feature_folder = 'data'
        self.label_folder = 'data'
        self.result_root = 'result'
        self.class_num = 2
        self.nan_rate = 0.3
        self.use_personality = True
        self.personality_K = 5
        self.is_probH = True
        self.m_prob = 1
        self.test_rate = 0.1
        self.normal_feature = True
        self.log = 'log/fc_valence.txt'
        self.gpu = 3
        self.print_to_screen = True

        # train
        self.n_epoch = 5000
        self.n_hid = 8
        self.device = 'cuda'
        self.lr = 0.001
        self.drop_out = 0.5
        self.weight_decay = 5e-4
        self.decay_step = 200
        self.decay_rate = 0.7
        self.milestone = [150]
        self.gamma = 1

        # test
        self.n_fold = 1
