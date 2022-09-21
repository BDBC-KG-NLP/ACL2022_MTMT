import sys
sys.path.append("..")
from mtmt_main.other.utils import cpu_2_gpu, eval_F1
from mtmt_main.other.productor import product_criterion
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from mtmt_main.other.config import ix_to_tag, tag_to_ix
from mtmt_main.trainers.base_trainer import BaseTrainer
from mtmt_main.saved_models.recognizer_saver import RecognizerSaver

from mtmt_main.models.evaluator_model import EvaluatorModel
from mtmt_main.dataloaders.evaluator_loader import SiameseLoader
from mtmt_main.models.recognizer_model import RecognizerModel
from mtmt_main.saved_models.evaluator_saver import EvaluatorSaver


class MTMT(BaseTrainer):
    def __init__(self, hp):
        # initialize the experiment
        super(MTMT, self).__init__(hp)

        self.tgt_lang = hp.tgt_lang

        self.tgt_train_set_dict = {
            'es': 'data/Conll/es/train.txt',
            'nl': 'data/Conll/nl/train.txt',
            'de': 'data/Conll/de/train.txt',
            'ar': 'data/WikiAnn/ar/train.txt',
            'hi': 'data/WikiAnn/hi/train.txt',
            'zh': 'data/WikiAnn/zh/train.txt',
        }

        self.target_trainset = self.tgt_train_set_dict[hp.tgt_lang]

        self.evaluator_model_path = EvaluatorSaver().models_dict[hp.tgt_lang]
        self.evaluator_state_dict = torch.load(self.evaluator_model_path)

        self.evaluator_model_name = 'EvaluatorModel'
        self.evaluator_model = cpu_2_gpu(EvaluatorModel(hp.batch_size, hp.LEN_ALL_TAGS, hp.HIDDEN_DIM, self.feature_dim))
        self.evaluator_model.load_state_dict(self.evaluator_state_dict)

        self.recognizer_model_path = RecognizerSaver().models_dict[hp.tgt_lang]
        self.recognizer_state_dict = torch.load(self.recognizer_model_path)

        self.recognizer_model_name = 'RecognizerModel'
        self.recognizer_model = cpu_2_gpu(RecognizerModel(hp.LEN_ALL_TAGS, hp.HIDDEN_DIM))
        self.recognizer_model.load_state_dict(self.recognizer_state_dict)

        self.student_model = cpu_2_gpu(RecognizerModel(hp.LEN_ALL_TAGS, hp.HIDDEN_DIM))

        self.alpha = hp.loss_alpha
        self.softmax = nn.Softmax(dim=-1)
        self.cossim = nn.CosineSimilarity(dim=1, eps=1e-6)

        linear_params = list(map(id, self.student_model.linear.parameters()))
        base_params = filter(lambda p: id(p) not in linear_params,
                             self.student_model.parameters())

        self.optimizer = optim.Adam([
            {'params': base_params},
            {'params': self.student_model.linear.parameters(), 'lr': hp.lr / 1000}], lr=hp.lr)

        self.weight_loss = hp.weight_loss
        if (self.weight_loss):
            self.criterion_BCE = nn.BCELoss(reduction='none')
            self.criterion_CE = nn.CrossEntropyLoss(reduction='none')
            self.criterion_MSE = nn.MSELoss(reduction='none')
        else:
            self.criterion_BCE = nn.BCELoss()
            self.criterion_CE = nn.CrossEntropyLoss()
            self.criterion_MSE = product_criterion('MSELoss')

        self.trainset_params = '500000_T_T_F'
        self.validset_params = '200000_T_F_T'
        self.testset_params = '200000_T_F_T'
        self.dataloader_name = 'SiameseLoader'

        self.trainloader_tgt = self.init_dataloader(self.target_trainset, self.dataloader_param_generator(self.trainset_params))
        self.validloader_sim = self.init_dataloader(self.validset, self.dataloader_param_generator(self.validset_params))
        self.testloader_sim = self.init_dataloader(self.testset, self.dataloader_param_generator(self.testset_params))

    def init_dataloader(self, fpath, params):
        dataset = SiameseLoader(fpath, params)
        data_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=params['shuffle'], num_workers=4,
                                 collate_fn=dataset.pad)
        return data_loader

    def dataloader_param_generator(self, param_name):
        values = param_name.split('_')
        param_dict = {
            'maxSamples': int(values[0]),
            'isequal': True if values[1] == 'T' else False,
            'shuffle': True if values[2] == 'T' else False,
            'istest': True if values[3] == 'T' else False
        }
        return param_dict

    def train(self, evaluator_model, recognizer_model, cls_model_stu, target_trainloader, i):
        cls_model_stu.train()
        evaluator_model.eval()
        recognizer_model.eval()
        start_t = time.time()
        loss_train, mseloss1_train, mseloss2_train, bceloss_train = [], [], [], []
        for idx, batch in enumerate(target_trainloader):
            # prepare input data
            words_1_2, wordpiece_idx_1_2, tag_1_2, x_1_2, y, att_mask_1_2, seqlen_1_2, word_idx_1_2 = batch
            seqlen_1, seqlen_2 = seqlen_1_2[0], seqlen_1_2[1]
            wordpiece_idx_1, wordpiece_idx_2 = wordpiece_idx_1_2[0], wordpiece_idx_1_2[1]
            tag_1, tag_2 = tag_1_2[0], tag_1_2[1]
            x_1, x_2, att_mask_1, att_mask_2, y = cpu_2_gpu(x_1_2 + att_mask_1_2 + [y])
            # tea model output data
            recognizer_logits_1, recognizer_pred_1, recognizer_embeds_1 = recognizer_model(x_1, att_mask_1, seqlen_1)
            recognizer_logits_2, recognizer_pred_2, recognizer_embeds_2 = recognizer_model(x_2, att_mask_2, seqlen_2)
            evaluator_prediction, _, _, _, _, _, _ = evaluator_model([x_1, x_2], wordpiece_idx_1_2)
            # stu model output data
            student_logits_1, student_tag_pred_1, student_embeds_1 = cls_model_stu(x_1, att_mask_1, seqlen_1)
            student_logits_2, student_tag_pred_2, student_embeds_2 = cls_model_stu(x_2, att_mask_2, seqlen_2)

            recognizer_logits_1 = self.softmax(recognizer_logits_1)
            recognizer_logits_2 = self.softmax(recognizer_logits_2)
            student_logits_1 = self.softmax(student_logits_1)
            student_logits_2 = self.softmax(student_logits_2)
            # extract logits, embeds and tag correspond to wp_idx
            wordpiece_idx_1 = [wp_i + 1 for wp_i in wordpiece_idx_1]
            wordpiece_idx_2 = [wp_i + 1 for wp_i in wordpiece_idx_2]
            recognizer_logits_1 = recognizer_logits_1[list(range(recognizer_logits_1.size()[0])), wordpiece_idx_1]
            recognizer_logits_2 = recognizer_logits_2[list(range(recognizer_logits_2.size()[0])), wordpiece_idx_2]
            tag_pred_w1 = recognizer_pred_1[list(range(recognizer_pred_1.size()[0])), wordpiece_idx_1]
            tag_pred_w2 = recognizer_pred_2[list(range(recognizer_pred_2.size()[0])), wordpiece_idx_2]
            student_logits_1 = student_logits_1[list(range(student_logits_1.size()[0])), wordpiece_idx_1]
            student_logits_2 = student_logits_2[list(range(student_logits_2.size()[0])), wordpiece_idx_2]
            student_embeds_1 = student_embeds_1[list(range(student_embeds_1.size()[0])), wordpiece_idx_1]
            student_embeds_2 = student_embeds_2[list(range(student_embeds_2.size()[0])), wordpiece_idx_2]

            student_embeds_1, student_embeds_2 = cpu_2_gpu([student_embeds_1, student_embeds_2])

            # pred similarity score by student model
            student_similarity = evaluator_model.evaluator_layer(student_embeds_1, student_embeds_2)

            self.optimizer.zero_grad()

            student_logits_1, student_logits_2, recognizer_logits_1, recognizer_logits_2, tag_pred_w1, tag_pred_w2, student_similarity = cpu_2_gpu(
                [student_logits_1, student_logits_2, recognizer_logits_1, recognizer_logits_2, tag_pred_w1, tag_pred_w2, student_similarity])

            # calculate loss
            loss_ner_1 = self.criterion_MSE(student_logits_1.float(), recognizer_logits_1.float())
            loss_ner_2 = self.criterion_MSE(student_logits_2.float(), recognizer_logits_2.float())
            loss_sim = self.criterion_BCE(student_similarity, evaluator_prediction.detach())

            # add weighting
            if (self.weight_loss):
                loss_ner_1 = torch.sum(loss_ner_1, dim=-1)
                loss_ner_2 = torch.sum(loss_ner_2, dim=-1)

                confidence_1 = torch.max(recognizer_logits_1.detach(), dim=1).values
                weight_alpha_1 = confidence_1 ** 2
                confidence_2 = torch.max(recognizer_logits_2.detach(), dim=1).values
                weight_alpha_2 = confidence_2 ** 2

                one_like_weight = torch.ones_like(evaluator_prediction.detach())
                weight_beta = (evaluator_prediction.detach() * 2 - one_like_weight) ** 2 + 0.5
                weight_beta = torch.where(weight_beta > 1, one_like_weight, weight_beta)

                prediction_sim = self.cossim(recognizer_logits_1.detach(), recognizer_logits_2.detach())
                prediction_sim = torch.sigmoid(prediction_sim)
                weight_gamma = one_like_weight - torch.abs(prediction_sim - evaluator_prediction.detach())

                loss_ner_1 = torch.mean(torch.mul(torch.mul(weight_gamma, weight_alpha_1), loss_ner_1))
                loss_ner_2 = torch.mean(torch.mul(torch.mul(weight_gamma, weight_alpha_2), loss_ner_2))
                loss_sim = torch.mean(torch.mul(torch.mul(weight_gamma, weight_beta), loss_sim))


            loss = self.alpha * loss_sim + 1 / 2 * loss_ner_1 + 1 / 2 * loss_ner_2

            loss.backward()

            self.optimizer.step()

            loss_train.append(loss.item())
            mseloss1_train.append(loss_ner_1.item())
            mseloss2_train.append(loss_ner_2.item())
            bceloss_train.append(loss_sim.item())
            if idx % 1000 == 0:  # monitoring
                self.logger.info(
                    f"STEP: {idx}\tLOSS={round(np.mean(loss_train), 11)}\t\tRecognizer_1 LOSS={round(np.mean(mseloss1_train), 11)}\t\tRecognizer_2 LOSS={round(np.mean(mseloss2_train), 11)}\t\tEvaluator LOSS={round(np.mean(bceloss_train), 11)}\t\ttime: {(time.time() - start_t) / 60}")

        return np.mean(loss_train)



    def evaluation(self, student_model, target_trainloader, i):
        student_model.eval()
        start_t = time.time()
        Y, Y_HAT, Embeds_1 = [], [], []
        Words, Word_Idx_list = [], []
        pred_list, tag_list, embed_list_cls, embed_list_cls_stu, embed_list_sim = [], [], [], [], []
        with torch.no_grad():
            for idx, batch in enumerate(target_trainloader):
                # prepare input data
                words_1_2, wordpiece_idx_1_2, tag_1_2, x_1_2, y, att_mask_1_2, seqlen_1_2, word_idx_1_2 = batch
                words_1, words_2 = words_1_2[0], words_1_2[1]
                seqlen_1, seqlen_2 = seqlen_1_2[0], seqlen_1_2[1]
                word_idx_1, word_idx_2 = word_idx_1_2[0], word_idx_1_2[1]
                wordpiece_idx_1, wordpiece_idx_2 = wordpiece_idx_1_2[0], wordpiece_idx_1_2[1]
                Words.extend(words_1)
                Word_Idx_list.extend(word_idx_1)
                tag_1, tag_2 = tag_1_2[0], tag_1_2[1]
                x_1, x_2, att_mask_1, att_mask_2 = cpu_2_gpu(x_1_2 + att_mask_1_2)

                # model output data
                logits_1, tag_pred_1, embeds_cls_1_stu = student_model(x_1, att_mask_1, seqlen_1)

                tag_pred_1 = [t_i[wp_idx + 1] for t_i, wp_idx in
                              zip(tag_pred_1.cpu().numpy().tolist(), wordpiece_idx_1)]
                tag_pred_1 = [ix_to_tag[0] if p == 9 else ix_to_tag[p] for p in tag_pred_1]
                pred_list.extend([tag_to_ix[t] for t in tag_pred_1])
                tag_list.extend([t.item() for t in tag_1])

                Y.extend([ix_to_tag[t.item()] for t in tag_1])
                Y_HAT.extend(tag_pred_1)
                if idx % 1000 == 0:  # monitoring
                    self.logger.info(f"TEST STEP: {idx}\t\tTIME: {(time.time() - start_t) / 60}")

        self.logger.info(f"============Eval by conlleval:============")
        precision, recall, f1 = eval_F1(Y, Y_HAT, 'conlleval')
        self.logger.info("PRE=%.5f\t\tREC=%.5f\t\tF1=%.5f" % (precision, recall, f1))

        return f1

    def train_epoch(self, i):
        if (i == 1):
            self.record_result_dict['BASE_MAX_F1'] = self.evaluation(self.recognizer_model, self.testloader_sim, i)

        self.logger.info(f"======================MTMT KD STUDENT={i}=====================")
        self.train(self.evaluator_model, self.recognizer_model, self.student_model, self.trainloader_tgt, i)
        self.logger.info(f"======================VALID BY STUDENT MODEL={i}=====================")
        f1 = self.evaluation(self.student_model, self.validloader_sim, i)

        if (f1 > self.record_result_dict['VALID_MAX_F1']):
            torch.save(self.student_model.state_dict(), self.record_dir_dict['chk_dir'] + 'best_mtmt.pt')
            self.record_result_dict['VALID_MAX_F1'] = f1
            self.record_result_dict['VALID_MAX_EPOCH'] = i

        self.logger.info(f"Evaluator Teacher: {self.evaluator_model_path}")
        self.logger.info(f"Recognizer Teacher: {self.recognizer_model_path}")
        self.logger.info(f"Recognizer Teacher F1: {self.record_result_dict['BASE_MAX_F1']}")

        self.logger.info(
            f"Best Student: EPOCH_NUM={self.record_result_dict['VALID_MAX_EPOCH']}\t F1={self.record_result_dict['VALID_MAX_F1']}")
        self.logger.info(
            f"Best Student Model SAVED IN: {self.record_dir_dict['chk_dir']}best_mtmt.pt")

        self.logger.info(f"======================TEST BY STUDENT MODEL={i}=====================")
        self.evaluation(self.student_model, self.testloader_sim, i)

