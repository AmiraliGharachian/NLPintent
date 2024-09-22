import os
import pickle
import json


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def create_tokens_and_labels(id, sample):
    intent = sample['intent']
    utt = sample['utt']
    annot_utt = sample['annot_utt']
    tokens = utt.split()
    labels = []
    label = 'O'
    split_annot_utt = annot_utt.split()
    idx = 0
    BIO_SLOT = False
    while idx < len(split_annot_utt):
        if split_annot_utt[idx].startswith('['):
            label = split_annot_utt[idx].lstrip('[')
            idx += 2
            BIO_SLOT = True
        elif split_annot_utt[idx].endswith(']'):
            if split_annot_utt[idx - 1] == ":":
                labels.append("B-" + label)
                label = 'O'
                idx += 1
            else:
                labels.append("I-" + label)
                label = 'O'
                idx += 1
            BIO_SLOT = False
        else:
            if split_annot_utt[idx - 1] == ":":
                labels.append("B-" + label)
                idx += 1
            elif BIO_SLOT == True:
                labels.append("I-" + label)
                idx += 1
            else:
                labels.append("O")
                idx += 1

    if len(tokens) != len(labels):
        raise ValueError(f"Len of tokens, {tokens}, doesnt match len of labels, {labels}, "
                         f"for id {id} and annot utt: {annot_utt}")
    return tokens, labels, intent


def Read_Massive_dataset(massive_raw):
    sentences_tr, tags_tr, intent_tags_tr = [], [], []
    sentences_val, tags_val, intent_tags_val = [], [], []
    sentences_test, tags_test, intent_tags_test = [], [], []
    all_tags, all_intents = [], []

    for id, sample in enumerate(massive_raw):
        if sample['partition'] == 'train':
            tokens, labels, intent = create_tokens_and_labels(id, sample)
            sentences_tr.append(tokens)
            tags_tr.append(labels)
            intent_tags_tr.append(intent)
            all_tags += labels

        if sample['partition'] == 'dev':
            tokens, labels, intent = create_tokens_and_labels(id, sample)
            sentences_val.append(tokens)
            tags_val.append(labels)
            intent_tags_val.append(intent)
            all_tags += labels

        if sample['partition'] == 'test':
            tokens, labels, intent = create_tokens_and_labels(id, sample)
            sentences_test.append(tokens)
            tags_test.append(labels)
            intent_tags_test.append(intent)
            all_tags += labels

    all_tags = list(set(all_tags))

    allintents = intent_tags_tr + intent_tags_val + intent_tags_test
    all_intents = list(set(allintents))
    return sentences_tr, tags_tr, intent_tags_tr, sentences_val, tags_val, intent_tags_val, sentences_test, tags_test, intent_tags_test, all_tags, all_intents



def parse_ourData_newformat(dir, save_dir=None):
    # returns JSON object as
    # a dictionary
    massive_raw_en = []
    with open(dir, 'r') as f:
        for line in f:
            massive_raw_en.append(json.loads(line))

    # Closing file
    f.close()

    sentences_tr, tags_tr, intent_tags_tr, sentences_val, tags_val, intent_tags_val, sentences_test, tags_test, intent_tags_test, all_tags, all_intents = Read_Massive_dataset(massive_raw_en)
    Data={}

    Data["tr_inputs"], Data["tr_tags"], Data["tr_intents"] = sentences_tr, tags_tr, intent_tags_tr
    Data["val_inputs"], Data["val_tags"], Data["val_intents"] = sentences_val, tags_val, intent_tags_val
    Data["test_inputs"], Data["test_tags"], Data["test_intents"] = sentences_test, tags_test, intent_tags_test

    dict2 = {}
    dict_rev2 = {}
    inte2 = {}
    inte_rev2 = {}

    for i, tag in enumerate(all_tags):
        dict_rev2[tag] = i + 1
        dict2[i + 1] = tag
    print("Slots labels: ", all_tags)
    print("Number of Slots labels :", len(all_tags))

    for i, tag in enumerate(all_intents):
        inte_rev2[tag] = i + 1
        inte2[i + 1] = tag
    print("Intent labels: ", all_intents)
    print("Number of Intents labels :", len(all_intents))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if save_dir is not None:
        save_obj(Data, save_dir + '/Data')
        save_obj(dict2, save_dir + '/dict2')
        save_obj(dict_rev2, save_dir + '/dict_rev2')
        save_obj(inte2, save_dir + '/inte2')
        save_obj(inte_rev2, save_dir + '/inte_rev2')
        save_obj(all_tags, save_dir + '/alltags')



