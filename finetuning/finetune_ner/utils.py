import logging
import os
import pdb
import json
import numpy as np

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

from filelock import FileLock

from tqdm import tqdm
from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available


logger = logging.getLogger(__name__)


@dataclass
class InputExample:
    """
    A single training/test example for token classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """

    guid: str
    words: List[str]
    labels: Optional[List[str]]
    entity_labels: Optional[List[int]]


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None
    entity_type_ids: Optional[List[int]] = None

class Split(Enum):
    train = "train"
    dev = "devel"
    test = "test"


if is_torch_available():
    import torch
    from torch import nn
    from torch.utils.data.dataset import Dataset

    class NerDataset(Dataset):
        """
        This will be superseded by a framework-agnostic approach
        soon.
        """

        features: List[InputFeatures]
        pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index
        # Use cross entropy ignore_index as padding label id so that only
        # real label ids contribute to the loss later.

        def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            labels: List[str],
            model_type: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
            data_list='',
            eval_data_list='',
        ):
            # Load data features from cache or dataset file
            cached_features_file = os.path.join(
                data_dir, "multi_cached_{}_{}_{}".format(mode.value, tokenizer.__class__.__name__, str(max_seq_length)),
            )

            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.
            lock_path = cached_features_file + ".lock"
            with FileLock(lock_path):

                if os.path.exists(cached_features_file) and not overwrite_cache:
                    logger.info(f"Loading features from cached file {cached_features_file}")
                    self.features = torch.load(cached_features_file)
                else:
                    logger.info(f"Creating features from dataset file at {data_dir}")
                    examples = read_examples_from_file(data_dir, mode, data_list=data_list, eval_data_list=eval_data_list)
                    # TODO clean up all this to leverage built-in features of tokenizers
                    self.features = convert_examples_to_features(
                        examples,
                        labels,
                        max_seq_length,
                        tokenizer,
                        cls_token_at_end=bool(model_type in ["xlnet"]),
                        # xlnet has a cls token at the end
                        cls_token=tokenizer.cls_token,
                        cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
                        sep_token=tokenizer.sep_token,
                        sep_token_extra=False,
                        # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                        pad_on_left=bool(tokenizer.padding_side == "left"),
                        pad_token=tokenizer.pad_token_id,
                        pad_token_segment_id=tokenizer.pad_token_type_id,
                        pad_token_label_id=self.pad_token_label_id,
                    )
                    # logger.info(f"Saving features into cached file {cached_features_file}")
                    # torch.save(self.features, cached_features_file)
                
        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]


def read_examples_from_file(data_dir, mode: Union[Split, str], data_list='', eval_data_list='') -> List[InputExample]:
    split = Split

    if isinstance(mode, split):
        mode = mode.value

    guid_index = 1
    examples = []

    def gen_dataset(file_path, guid_index, examples, merge_idx=0, downsample=False, merge_data=""):
        if not os.path.exists(file_path):
            return examples, guid_index
        
        else:
            with open(file_path, encoding="utf-8") as f:
                words = []
                labels = []
                entity_labels = []
                new_ex = []
                for line_idx, line in enumerate(f):
                    if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                        if words:
                            new_ex.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels, entity_labels=entity_labels))
                            # examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels, entity_labels=entity_labels))
                            guid_index += 1
                            words = []
                            labels = []
                            entity_labels = []
                    else:
                        splits = line.split(" ")
                        words.append(splits[0])
                        entity_labels.append(merge_idx)
                        if len(splits) > 1:
                            splits_replace = splits[-1].replace("\n", "")
                            labels.append(splits_replace)
                        else:
                            # Examples could have no label for mode = "test"
                            labels.append("O")
                    
                if words:
                    new_ex.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels, entity_labels=entity_labels))
                    # examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels, entity_labels=entity_labels))
            
            if downsample:
                if merge_data in ["NCBI-disease", "BC4CHEMD", "BC2GM", "linnaeus", "JNLPBA-cl", "JNLPBA-ct", "JNLPBA-dna", "JNLPBA-rna"]:
                                #   "Biological_structure", "Diagnostic_procedure", "Duration", "Date", "Therapeutic_procedure", "Sign_symptom", 
                                #   "Lab_value"]:
                    examples.extend(new_ex)
                else:
                    if merge_idx == 1:
                        # random.shuffle(new_ex)
                        total_len = len(new_ex)
                        new_ex = new_ex[:int(total_len * 0.25)]
                    # elif merge_idx == 2:
                    #     # random.shuffle(new_ex)
                    #     total_len = len(new_ex)
                    #     new_ex = new_ex[:int(total_len * 0.4)]
                    elif merge_idx == 3:
                        # random.shuffle(new_ex)
                        total_len = len(new_ex)
                        new_ex = new_ex[:int(total_len * 0.25)]
                    elif merge_idx == 4:
                        # random.shuffle(new_ex)
                        total_len = len(new_ex)
                        new_ex = new_ex[:int(total_len * 0.62)]
                    # elif merge_idx == 5:
                    #     # random.shuffle(new_ex)
                    #     total_len = len(new_ex)
                    #     new_ex = new_ex[:int(total_len * 0.713)]
                    examples.extend(new_ex)
            else:
                examples.extend(new_ex)

            return examples, guid_index

    def get_merge_idx(data_name):
        if data_name in ["NCBI-disease", "BC5CDR-disease", "mirna-di", "ncbi_disease", "scai_disease", "variome-di"]:
            merge_idx = 1
        elif data_name in ["BC5CDR-chem",  "cdr-ch", "chemdner", "scai_chemicals", "chebi-ch", "BC4CHEMD"]:
            merge_idx = 2
        elif data_name in ["BC2GM", "JNLPBA-protein", "bc2gm", "mirna-gp", "cell_finder-gp", "chebi-gp", "loctext-gp", "deca", "fsu", "gpro", "jnlpba-gp", "bio_infer-gp", "variome-gp", "osiris-gp",  "iepa"]:
            merge_idx = 3
        elif data_name in ["s800", "linnaeus", "loctext-sp", "mirna-sp", "chebi-sp", "cell_finder-sp", "variome-sp"]:
            merge_idx = 4
        elif data_name in ["JNLPBA-cl", "cell_finder-cl", "jnlpba-cl", "gellus", "cll"]:
            merge_idx = 5
        elif data_name in ["JNLPBA-dna", "jnlpba-dna"]:
            merge_idx = 6
        elif data_name in ["JNLPBA-rna","jnlpba-rna"]:
            merge_idx = 7
        elif data_name in ["JNLPBA-ct","jnlpba-ct"]:
            merge_idx = 8
        else:
            merge_idx = 0
        return merge_idx

    if 'train' in mode:
        # prepare all data with batch-wise, epoch-wise, and all random
        data_list = data_list.split('+')
        for _, merge_data in enumerate(data_list):
            merge_idx = get_merge_idx(merge_data)
            file_path = os.path.join(data_dir+merge_data, f"{mode}.txt")
            examples, guid_index = gen_dataset(file_path, guid_index, examples, merge_idx=merge_idx, downsample=False)
            # examples, guid_index = gen_dataset(file_path, guid_index, examples, merge_idx=merge_idx, downsample=True, merge_data=merge_data)
        # random.shuffle(examples)
    else:
        # prepare specific data name with eval_data_type
        eval_data_list = eval_data_list.split('+')
        for _, eval_data_type in enumerate(eval_data_list):
            merge_idx = get_merge_idx(eval_data_type)
            file_path = os.path.join(data_dir+eval_data_type, f"{mode}.txt")
            examples, guid_index = gen_dataset(file_path, guid_index, examples, merge_idx=merge_idx)
        
    return examples


def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_seq_length: int,
    tokenizer: PreTrainedTokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
) -> List[InputFeatures]:
    """ Loads a data file into a list of `InputFeatures`
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    # TODO clean up all this to leverage built-in features of tokenizers

    label_map = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_map)
    features = []
    
    for (ex_index, example) in tqdm(enumerate(examples)):
        if ex_index % 10_000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens, label_ids = [], []

        for word_idx, (word, label) in enumerate(zip(example.words, example.labels)):
            word_tokens = tokenizer.tokenize(word)
            
            # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = tokenizer.num_special_tokens_to_add()
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]
            
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            entity_type_ids += [example.entity_labels[0]]
            
        # make entity type label index for multiner
        entity_type_ids = [example.entity_labels[0]] * len(tokens)
        segment_ids = [sequence_a_segment_id] * len(tokens)
        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
            entity_type_ids += [example.entity_labels[0]]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids
            entity_type_ids = [example.entity_labels[0]] + entity_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
            entity_type_ids = ([example.entity_labels[0]] * padding_length) + entity_type_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length
            entity_type_ids += [example.entity_labels[0]] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(entity_type_ids) == max_seq_length

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
            logger.info("entity_type_ids: %s", " ".join([str(x) for x in entity_type_ids]))

        if "token_type_ids" not in tokenizer.model_input_names:
            segment_ids = None
        
        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, \
                label_ids=label_ids, entity_type_ids=entity_type_ids,
            )
        )

    return features


def get_bio_labels(path: str) -> List[str]:
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return ["O", "B", "I"]

def get_labels(path: str) -> List[str]:
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
            
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

