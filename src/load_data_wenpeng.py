

import codecs
import glob
import io
import os
from torchtext import data
# from torchtext.data.datasets

def load_snli(text_field, label_field):
    print('Using load_snli wenpeng version')

    fields = {'sentence1': ('premise', text_field),
            'sentence2': ('hypothesis', text_field),
            'gold_label': ('label', label_field)}

    new_fields = {'premise': text_field,
            'hypothesis': text_field,
            'label': label_field}

    examples = []
    train_read = codecs.open('/save/wenpeng/datasets/StanfordEntailment/train.txt', 'r', 'utf-8')
    for line in train_read:
        parts = line.strip().split('\t')
        label = int(parts[0])
        sent1 = parts[1]
        sent2 = parts[2]
        dict_line = {'gold_label':label, 'sentence1':sent1, 'sentence2': sent2}
        examples.append(data.Example.fromdict(dict_line, fields))
    train_read.close()
    train_set = data.Dataset(examples, new_fields)

    examples = []
    dev_read = codecs.open('/save/wenpeng/datasets/StanfordEntailment/dev.txt', 'r', 'utf-8')
    for line in dev_read:
        parts = line.strip().split('\t')
        label = int(parts[0])
        sent1 = parts[1]
        sent2 = parts[2]
        dict_line = {'gold_label':label, 'sentence1':sent1, 'sentence2': sent2}
        examples.append(data.Example.fromdict(dict_line, fields))
    dev_read.close()
    dev_set = data.Dataset(examples, new_fields)

    examples = []
    test_read = codecs.open('/save/wenpeng/datasets/StanfordEntailment/test.txt', 'r', 'utf-8')
    for line in test_read:
        parts = line.strip().split('\t')
        label = int(parts[0])
        sent1 = parts[1]
        sent2 = parts[2]
        dict_line = {'gold_label':label, 'sentence1':sent1, 'sentence2': sent2}
        examples.append(data.Example.fromdict(dict_line, fields))
    test_read.close()
    test_set = data.Dataset(examples, new_fields)
    return train_set, dev_set, test_set
