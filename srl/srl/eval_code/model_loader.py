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

from srl_custom_dataset import CustomSrlReader
from srl_custom_model import CustomSemanticRoleLabeler, write_to_conll_eval_file, write_bio_formatted_tags_to_file, write_conll_formatted_tags_to_file, convert_bio_tags_to_conll_format

from custom_span_based_f1_measure import CustomSpanBasedF1Measure
from allennlp.models.archival import archive_model, load_archive, CONFIG_NAME
from allennlp.data.iterators import BucketIterator
from allennlp.training.metrics import DecodeSpanBasedF1Measure

from helper import get_viterbi_pairwise_potentials, decode


def load_model(archive_dir, weight_file, gpu_device):
    weight_file = os.path.join(archive_dir, weight_file)
    archive = load_archive(archive_dir, cuda_device=-1, weights_file=weight_file)
    model = archive.model

    model.eval()
    return model

def load_dataset(model, archive_dir, weight_file, batch_size, valid_data=False):
    config_file = os.path.join(archive_dir, "config_loss.json")
    params = Params.from_file(config_file)

    # Is this required??
    # params = archive.config 

    label_encoding="BIO"
    try:
            label_encoding = params["dataset_reader"]["label_encoding"]
    except:
            pass    

    print("Label encoding!", label_encoding)

    validation_iterator_params = params["iterator"]
    # validation_iterator = DataIterator.from_params(validation_iterator_params)
    validation_iterator = BucketIterator(batch_size=batch_size, sorting_keys=[("tokens", "num_tokens")])
    validation_iterator.index_with(model.vocab)

    dataset_name = 'train_data_path'
    if(valid_data):
            dataset_name = 'validation_data_path'

    print(f"Dataset: {dataset_name}")

    validation_and_test_dataset_reader = DatasetReader.from_params(params['dataset_reader'])
    validation_data_path = params[dataset_name]
    validation_data = validation_and_test_dataset_reader.read(validation_data_path)

    return validation_data, validation_iterator