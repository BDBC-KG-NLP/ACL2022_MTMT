import time
import sys
sys.path.append("..")
from mtmt_main.other.utils import cpu_2_gpu, eval_F1
from mtmt_main.other.productor import product_optimizer, product_criterion
from mtmt_main.other.custom_error import CustomError
from mtmt_main.other.config import ix_to_tag
from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch
from mtmt_main.trainers.base_trainer import BaseTrainer
from mtmt_main.dataloaders.recognizer_loader import RecognizerLoader
from mtmt_main.models.recognizer_model import RecognizerModel


class Recognizer(BaseTrainer):
    def __init__(self, hp):
        super(Recognizer, self).__init__(hp)

        # build model, reload potential checkpoints
        self.model_name, self.optim_name, self.criterion_name = 'RecognizerModel', "Adam", "CrossEntropyLoss"
        self.model = cpu_2_gpu(RecognizerModel(hp.LEN_ALL_TAGS, hp.HIDDEN_DIM))


        self.optimizer = product_optimizer(self.optim_name, self.model.parameters(), hp.lr)
        self.criterion = product_criterion(self.criterion_name)


        # load data
        self.dataloader_name = "RecognizerLoader"
        self.tgt_lang = hp.tgt_lang

        if(self.tgt_lang in ['es','nl','de']):
            self.data_dir = 'data/Conll/en/'
        elif(self.tgt_lang in ['ar','hi','zh']):
            self.data_dir = 'data/WikiAnn/en/'
        else:
            raise CustomError(f'tgt_lang {self.tgt_lang} is error!', -1)
            sys.exit()
        self.trainset = self.data_dir + 'train.txt'


        self.trainloader = self.init_dataloader(self.trainset, {'shuffle': True})
        self.validloader = self.init_dataloader(self.validset, {'shuffle': False})
        self.testloader = self.init_dataloader(self.testset, {'shuffle': False})


    def init_dataloader(self, fpath, params):
        dataset = RecognizerLoader(fpath)
        data_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=params['shuffle'], num_workers=4,
                                 collate_fn=dataset.pad)
        return data_loader

    def train(self, model, trainloader, optimizer, criterion):
        model.train()
        loss_train = []
        supconloss_train, celoss_train = [], []
        start_t = time.time()

        for idx, batch in enumerate(trainloader):
            words, x, is_heads, tags, y, seqlens, is_entity = batch
            x, y, is_heads = cpu_2_gpu([x, y, is_heads])

            _y = y  # for monitoring
            optimizer.zero_grad()
            logits, prediction, embeds = model(x, is_heads, seqlens)

            logits = logits.view(-1, logits.shape[-1])
            _y = _y.view(-1)
            is_heads = is_heads.view(-1)
            logits = logits[is_heads==1]
            _y = _y[is_heads==1]

            loss = criterion(logits, _y)
            loss_train.append(loss.item())
            loss.backward()
            optimizer.step()

            if idx % 100 == 0:  # monitoring
                self.logger.info(
                    f"STEP: {idx}\tLOSS={round(np.mean(loss_train), 11)}\t\ttime: {(time.time() - start_t) / 60}")

        return np.mean(loss_train)


    def evaluation(self, model, validloader):
        model.eval()
        Words, Is_heads, Tags, Y, Y_hat, Y_hat_List = [], [], [], [], [], []

        with torch.no_grad():
            for idx, batch in enumerate(validloader):
                words, x, is_heads, tags, y, seqlens, is_entity = batch
                x, y, is_heads = cpu_2_gpu([x, y, is_heads])
                _y = y
                logits, prediction, embeds = model(x, is_heads, seqlens)
                Words.extend(words)
                Is_heads.extend(is_heads.cpu().numpy().tolist())
                Tags.extend(tags)
                Y_hat_List.extend(prediction.cpu().numpy().tolist())

                for t_i, pred, heads, embed in zip(_y.cpu().numpy().tolist(), prediction.cpu().numpy().tolist(), is_heads.cpu().numpy().tolist(), embeds.cpu().numpy().tolist()):
                    Y.extend([t for head, t in zip(heads, t_i) if head == 1][1:-1])
                    pred = [0 if p == 9 else p for p in pred]
                    Y_hat.extend([p for head, p in zip(heads, pred) if head == 1][1:-1])


        self.logger.info(f"============Eval Recognizer by Conlleval:============")
        precision, recall, f1 = eval_F1([ix_to_tag[t_ix] for t_ix in Y], [ix_to_tag[t_ix] for t_ix in Y_hat], 'conlleval')
        self.logger.info("PRE=%.5f\t\tREC=%.5f\t\tF1=%.5f" % (precision, recall, f1))

        return precision, recall, f1

    def train_epoch(self, i):
        # train set : labeled data: english
        self.logger.info(f"=========================TRAIN AT EPOCH={i}=========================")
        t_loss = self.train(self.model, self.trainloader, self.optimizer, self.criterion)
        self.record_result_dict['train_loss_list'].append(t_loss)

        # valid set : unlabeled data: es、nl、de ...
        self.logger.info(f"=========================DEV AT EPOCH={i}===========================")
        precision, recall, f1 = self.evaluation(self.model, self.validloader)
        self.record_result_dict['dev_F1_list'].append(f1)

        if (self.record_result_dict['VALID_MAX_F1'] < f1):
            self.logger.info(f"Best MODEL UPDATE FROM {self.record_result_dict['VALID_MAX_F1']} to {f1}")
            self.record_result_dict['VALID_MAX_F1'], self.record_result_dict['VALID_MAX_EPOCH'] = f1, i
            torch.save(self.model.state_dict(), self.record_dir_dict['chk_dir'] + 'best_recognizer.pt')

        self.logger.info(f"Best Valid F1: EPOCH_NUM={self.record_result_dict['VALID_MAX_EPOCH']}\tF1_MAX={self.record_result_dict['VALID_MAX_F1']}")
        self.logger.info(f"Best MODEL SAVED in {self.record_dir_dict['chk_dir']}best_recognizer.pt")

        # test set : unlabeled data: es、nl、de ...
        self.logger.info(f"=========================TEST AT EPOCH={i}==========================")
        precision, recall, f1 = self.evaluation(self.model, self.testloader)
        self.record_result_dict['test_F1_list'].append(f1)


