import os, sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from seqTagger import transformertagger
from BERT_Joint_IDSF import BertIDSF
import project_statics
from utils import load_obj
from seqeval.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score
import torch
import time


if __name__ == '__main__':

    # now we use it
    save_path = 'test_IDSF_bert'
    data_path = project_statics.SFID_pickle_files
    Data = load_obj(data_path + '/Data')
    dict2 = load_obj(data_path + '/dict2')
    inte2 = load_obj(data_path + '/inte2')

    tagger_obj = transformertagger(save_path, BertIDSF, dict2, inte2, device=torch.device("cpu"))

        # calculating test results
    test_texts = []
    start_time = time.time()
    toks, predicted_labels, predicted_intents = tagger_obj.get_label(Data["test_inputs"], need_tokenization=False)
    end_time = time.time()
    print("Required time to calculate intent and slot labels for all test samples: ", end_time - start_time)

    true_labels = Data["test_tags"]
    true_intents = Data["test_intents"]

    print("Test Slots Recall: ", recall_score(true_labels, predicted_labels))
    print("Test Slots Precision: ", precision_score(true_labels, predicted_labels))
    print("Test Slots F1: ", f1_score(true_labels, predicted_labels))
    print("Test Intents Accuracy: ", accuracy_score(true_intents, predicted_intents))
    EM = 0
    for i in range(len(predicted_labels)):
        if accuracy_score(true_labels[i], predicted_labels[i]) == 1 and true_intents[i] == predicted_intents[i]:
            EM += 1
    print('Test Sentence Accuracy: ', EM / len(predicted_labels))


    # now we use it
    start_time1 = time.time()
    toks, labels, intents = tagger_obj.get_label(["hey what has been happening interesting in sports lately", "tell me some best tourist places to visit in america"], need_tokenization=True)
    end_time1 = time.time()
    print("Required time to calculate intent and slot labels for 2 samples: ",end_time1-start_time1)

    print("Golden Labels\n")
    print("hey:O what:O has:O been:O happening:O interesting:O in:O sports:O lately:O <=> general_quirky")
    print("tell:O me:O some:O best:B-place_name tourist:I-place_name places:I-place_name to:O visit:O in:O america:B-place_name <=> recommendation_locations")

    for i in range(len(toks)):
      print("\n## Intent:", intents[i])
      print("## Slots:")
      for token, slot in zip(toks[i],  labels[i]):
          print(f"{token:>10} : {slot}")



