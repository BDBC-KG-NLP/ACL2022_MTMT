import argparse
import sys
sys.path.append("..")
import os
from mtmt_main.other.config import *
from tqdm import tqdm
from mtmt_main.trainers.recognizer_trainer import Recognizer
from mtmt_main.trainers.evaluator_trainer import Evaluator
from mtmt_main.trainers.mtmt_trainer import MTMT


def config_parser():
    parser = argparse.ArgumentParser()
    # hyper params
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--loss_alpha", type=float, default=0.5)
    parser.add_argument("--LEN_ALL_TAGS", type=int, default=len(ALL_TAGS))
    parser.add_argument("--HIDDEN_DIM", type=int, default=HIDDEN_DIM)
    parser.add_argument("--weight_loss", dest="weight_loss", action="store_true")

    # model config
    parser.add_argument("--tgt_lang", type=str, default="es")
    parser.add_argument("--trainer_name", type=str, default="Recognizer")



    return parser

def main():
    # build trainer, reload potential checkpoints / build evaluator
    trainer = eval(hp.trainer_name)(hp)

    # training
    for i in tqdm(range(1, hp.n_epochs + 1, 1)):
        # train set : labeled data: english
        trainer.train_epoch(i)


if __name__ == '__main__':
    # parse parameters
    parser = config_parser()
    hp = parser.parse_args()

    # check parameters

    # run experiment
    main()


