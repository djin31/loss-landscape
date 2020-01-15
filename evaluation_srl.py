from IPython.core.debugger import Pdb
import os
import sys

sys.path.append('../code')
from typing import Iterator, List, Dict
import torch
import torch.optim as optim
import numpy as np

from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding, TokenCharactersEncoder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.modules.seq2vec_encoders import CnnEncoder

from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator

#from allennlp.training.learning_rate_schedulers import LearningRateWithMetricsWrapper
from allennlp.predictors import SentenceTaggerPredictor

from torch.optim.lr_scheduler import ReduceLROnPlateau
from allennlp.nn import util as nn_util

from allennlp.common.params import Params
from allennlp.training import util as training_util
from allennlp.common import util as common_util
from allennlp.models.model import Model, _DEFAULT_WEIGHTS
#from custom_trainer import CustomSrlTrainer, CustomSrlTrainerPieces
import argparse
import datetime
import logging
import shutil
import time
import re
import datetime
import traceback

def eval_loss(net, loader, use_cuda=False, gpu_device=0):
    if use_cuda:
        net=  net.cuda()
    net.eval()
    
    loss = 0 
    batches = 0
    
    validation_data = loader[0]
    validation_iterator = loader[1]
    val_generator = validation_iterator(validation_data, num_epochs=1, shuffle=False)
    with torch.no_grad():
        for batch in val_generator:
            batches+=1
            if use_cuda:
                batch = nn_util.move_to_device(batch,gpu_device)
            scores = net(
                        tokens=batch["tokens"], 
                        verb_indicator=batch["verb_indicator"], 
                        tags=batch["tags"], 
                        metadata=batch["metadata"]
                    )
            loss+=scores["loss"].item()

    return loss/batches, (1-loss)/batches
