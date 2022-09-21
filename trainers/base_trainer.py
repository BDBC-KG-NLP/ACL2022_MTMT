import sys
sys.path.append("..")

from mtmt_main.other.utils import get_identify_name, get_logger
from mtmt_main.other.productor import create_all_record_dir

class BaseTrainer(object):
    def __init__(self, hp):
        # initialize the experiment
        self.identify_name = get_identify_name(hp)
        self.record_dirs = {
            'plot_dir': '../MassPlotPng/',
            'tboard_dir': '../MassLogBoard/',
            'result_dir': '../MassPredictResult/',
            'pt_dir': '../MassPT/',
            'logger_dir': '../MassLogInfo/',
            'chk_dir': '../checkpoints/'
        }
        self.record_dir_dict = create_all_record_dir(self.record_dirs, hp)
        self.logger = get_logger('info', self.record_dir_dict['logger_dir'] + 'log')
        self.record_result_dict = {
            'train_loss_list': [],
            'dev_F1_list': [],
            'test_F1_list': [],
            'BASE_MAX_F1': float("-inf"),
            'VALID_MAX_PREC': float("-inf"),
            'VALID_MAX_REC': float("-inf"),
            'VALID_MAX_F1': float("-inf"),
            'VALID_MAX_EPOCH': float("-inf"),
            'TEST_MAX_PREC': float("-inf"),
            'TEST_MAX_REC': float("-inf"),
            'TEST_MAX_F1': float("-inf"),
            'TEST_MAX_EPOCH': float("-inf"),
        }
        self.batch_size = hp.batch_size
        self.feature_dim = 768

        self.dataset_dict = {
            'es': 'data/Conll/es/',
            'nl': 'data/Conll/nl/',
            'de': 'data/Conll/de/',
            'ar': 'data/WikiAnn/ar/',
            'hi': 'data/WikiAnn/hi/',
            'zh': 'data/WikiAnn/zh/',
        }
        self.validset = self.dataset_dict[hp.tgt_lang] + 'valid.txt'
        self.testset = self.dataset_dict[hp.tgt_lang] + 'test.txt'

    def train(self):
        pass
    def evaluateion(self):
        pass
