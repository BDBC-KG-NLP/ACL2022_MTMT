import time
import sys
sys.path.append("..")
from torch.utils.data import Dataset, DataLoader
from mtmt_main.other.utils import cpu_2_gpu, eval_F1
from mtmt_main.other.productor import product_optimizer, product_criterion
from mtmt_main.other.custom_error import CustomError
import numpy as np
import torch

from mtmt_main.trainers.base_trainer import BaseTrainer
from mtmt_main.models.evaluator_model import EvaluatorModel
from mtmt_main.dataloaders.evaluator_loader import SiameseLoader

class Evaluator(BaseTrainer):
    def __init__(self, hp):
        # initialize the experiment
        super(Evaluator, self).__init__(hp)

        # build model, reload potential checkpoints
        self.model_name = 'EvaluatorModel'
        self.model = cpu_2_gpu(EvaluatorModel(hp.batch_size, hp.LEN_ALL_TAGS, hp.HIDDEN_DIM, self.feature_dim))
        self.optimizer = product_optimizer('Adam', self.model.parameters(), hp.lr)
        self.criterion = product_criterion('BCELoss')

        # load data
        self.dataloader_name = "SiameseLoader"
        self.trainset_params = "500000_T_T_F"
        self.validset_params = "500000_T_F_F"
        self.testset_params = "500000_T_F_F"
        self.tgt_lang = hp.tgt_lang

        if (self.tgt_lang in ['es', 'nl', 'de']):
            self.data_dir = 'data/Conll/en/'
        elif (self.tgt_lang in ['ar', 'hi', 'zh']):
            self.data_dir = 'data/WikiAnn/en/'
        else:
            raise CustomError(f'tgt_lang {self.tgt_lang} is error!', -1)
            sys.exit()
        self.trainset = self.data_dir + 'train.txt'

        self.trainloader = self.init_dataloader(self.trainset, self.dataloader_param_generator(self.trainset_params))
        self.validloader = self.init_dataloader(self.validset, self.dataloader_param_generator(self.validset_params))
        self.testloader = self.init_dataloader(self.testset, self.dataloader_param_generator(self.testset_params))


    def dataloader_param_generator(self, param_name):
        values = param_name.split('_')
        param_dict = {
            'maxSamples': int(values[0]),
            'isequal': True if values[1] == 'T' else False,
            'shuffle': True if values[2] == 'T' else False,
            'istest': True if values[3] == 'T' else False
        }
        return param_dict

    def init_dataloader(self, fpath, params):
        dataset = SiameseLoader(fpath, params)
        data_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=params['shuffle'], num_workers=4,
                                 collate_fn=dataset.pad)
        return data_loader

    def train(self, model, trainloader, optimizer, criterion):
        model.train()
        loss_train = []
        start_t = time.time()
        for idx, batch in enumerate(trainloader):
            words_1_2, wordpiece_idx_1_2, tag_1_2, x_1_2, y, att_mask_1_2, seqlen_1_2, word_idx_1_2 = batch
            x_1, x_2, y, tag_1, tag_2 = cpu_2_gpu(x_1_2+[y]+tag_1_2)
            optimizer.zero_grad()
            sim, _, _, _, _, _, _ = model([x_1, x_2], wordpiece_idx_1_2)
            loss = criterion(sim, y.float())
            loss.backward()
            optimizer.step()

            loss_train.append(loss.item())
            if idx % 100 == 0:
                self.logger.info(
                    f"TRAIN STEP: {idx}\tLOSS_MAIN={round(np.mean(loss_train), 11)}\t\tTIME: {(time.time() - start_t) / 60}")
        return np.mean(loss_train)

    def evaluation(self, model, validloader):
        model.eval()
        start_t = time.time()
        Words_1, Is_heads_1, Tags_1, Word_Idx_1, Words_2, Is_heads_2, Tags_2, Word_Idx_2, Y, Y_hat = [], [], [], [], [], [], [], [], [], []
        embed_list_cls, tag_list = [], []
        with torch.no_grad():
            for idx, batch in enumerate(validloader):
                words_1_2, wordpiece_idx_1_2, tag_1_2, x_1_2, y, att_mask_1_2, seqlen_1_2, word_idx_1_2 = batch
                words_1, words_2 = words_1_2[0], words_1_2[1]
                word_idx_1, word_idx_2 = word_idx_1_2[0], word_idx_1_2[1]
                att_mask_1, att_mask_2 = att_mask_1_2[0], att_mask_1_2[1]

                x_1, x_2, y, tag_1, tag_2 = cpu_2_gpu(x_1_2 + [y] + tag_1_2)
                prediction, _, _, _, _, embeds_1, _ = model([x_1, x_2], wordpiece_idx_1_2)
                embed_list_cls.extend(embeds_1.cpu().numpy().tolist())
                tag_list.extend(tag_1.cpu().numpy().tolist())

                Words_1.extend(words_1)
                Words_2.extend(words_2)
                Is_heads_1.extend(att_mask_1.cpu().numpy())
                Is_heads_2.extend(att_mask_2.cpu().numpy())
                Word_Idx_1.extend(word_idx_1)
                Word_Idx_2.extend(word_idx_2)
                Tags_1.extend(tag_1)
                Tags_2.extend(tag_2)
                Y.extend(y.cpu().numpy().tolist())
                Y_hat.extend(prediction.cpu().numpy().tolist())

                if idx % 1000 == 0:  # monitoring
                    self.logger.info(f"Siamese STEP: {idx}\t\ttime: {(time.time() - start_t) / 60}")

        self.logger.info(f"============Eval by sklearn_f1:============")
        Y_hat = [1 if y_pred >= 0.5 else 0 for y_pred in Y_hat]
        precision, recall, f1 = eval_F1(Y, Y_hat, 'sklearn_f1')
        self.logger.info("PRE=%.5f\t\tREC=%.5f\t\tF1=%.5f" % (precision, recall, f1))

        return precision, recall, f1

    def train_epoch(self, i):
        # train set : labeled data: english
        self.logger.info(f"=========================TRAIN AT EPOCH={i}=========================")
        t_loss = self.train(self.model, self.trainloader, self.optimizer, self.criterion)
        self.record_result_dict['train_loss_list'].append(t_loss)

        # valid set : unlabeled data: es、nl、de、ch...
        self.logger.info(f"=========================DEV AT EPOCH={i}===========================")
        precision, recall, f1 = self.evaluation(self.model, self.validloader)
        self.record_result_dict['dev_F1_list'].append(f1)

        if (self.record_result_dict['VALID_MAX_F1'] < f1):
            self.record_result_dict['VALID_MAX_F1'], self.record_result_dict['VALID_MAX_EPOCH'] = f1, i
            torch.save(self.model.state_dict(), self.record_dir_dict['chk_dir'] + 'best_evaluator.pt')

        self.logger.info(
            f"Best F1: EPOCH_NUM={self.record_result_dict['VALID_MAX_EPOCH']}\tF1_MAX={self.record_result_dict['VALID_MAX_F1']}")
        self.logger.info(f"Best Model Saved in {self.record_dir_dict['chk_dir']}best_evaluator.pt")

        # test set : unlabeled data: es、nl、de、ch...
        self.logger.info(f"=========================TEST AT EPOCH={i}==========================")
        precision, recall, f1 = self.evaluation(self.model, self.testloader)
        self.record_result_dict['test_F1_list'].append(f1)


