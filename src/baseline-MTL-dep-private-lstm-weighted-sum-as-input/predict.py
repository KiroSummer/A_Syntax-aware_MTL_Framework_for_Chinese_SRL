''' Predict and output scores.

   - Reads model param file.
   - Runs data.
   - Remaps label indices.
   - Outputs protobuf file.
'''

from neural_srl.shared import *
from neural_srl.shared.constants import *
from neural_srl.shared.dictionary import Dictionary
from neural_srl.shared.tagger_data import TaggerData
from neural_srl.shared.measurements import Timer
from neural_srl.pytorch.tagger import BiLSTMTaggerModel
from neural_srl.shared.inference_utils import *
from neural_srl.shared.srl_eval_utils import *
from neural_srl.shared.reader import load_eval_data
# from neural_srl.shared.numpy_saver import NumpySaver

import argparse
import numpy
import os
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',
                        type=str,
                        default='',
                        required=True,
                        help='Path to the model directory.')

    parser.add_argument('--input',
                        type=str,
                        default='',
                        required=True,
                        help='Path to the input file path (sequetial tagging format).')

    parser.add_argument('--task',
                        type=str,
                        help='Training task (srl or propid). Default is srl.',
                        default='srl',
                        choices=['srl', 'propid'])

    parser.add_argument('--gold',
                        type=str,
                        default='',
                        help='(Optional) Path to the file containing gold propositions (provided by CoNLL shared task).')

    parser.add_argument('--inputprops',
                        type=str,
                        default='',
                        help='(Optional) Path to the predicted predicates in CoNLL format. Ignore if using gold predicates.')

    parser.add_argument('--output',
                        type=str,
                        default='',
                        help='(Optional) Path for output predictions.')

    parser.add_argument('--outputprops',
                        type=str,
                        default='',
                        help='(Optional) Path for output predictions in CoNLL format. Only used when task is {propid}.')

    parser.add_argument('--proto',
                        type=str,
                        default='',
                        help='(Optional) Path to the proto file path (for reusing predicted scores).')
    parser.add_argument('--gpu',
                        type=str,
                        default="",
                        help='(Optional) A argument that specifies the GPU id. Default use the cpu')

    args = parser.parse_args()
    config = configuration.get_config(os.path.join(args.model, 'config'))

    # Detect available ensemble models.
    task = "srl"
    num_ensemble_models = 1
    if num_ensemble_models == 1:
        print ('Using single model.')
    else:
        print ('Using an ensemble of {} models'.format(num_ensemble_models))

    ensemble_scores = None
    for i in range(num_ensemble_models):
        if num_ensemble_models == 1:
            model_path = os.path.join(args.model, 'model')
            word_dict_path = os.path.join(args.model, 'word_dict')
        else:
            model_path = os.path.join(args.model, 'model{}.npz'.format(i))
            word_dict_path = os.path.join(args.model, 'word_dict{}'.format(i))
        head_dict_path = os.path.join(args.model, 'head_dict')
        char_dict_path = os.path.join(args.model, 'char_dict')
        label_dict_path = os.path.join(args.model, 'label_dict')
        dep_label_path = os.path.join(args.model, 'dep_label_dict')
        print("model dict path: {}, word dict path: {}, head dict path: {}, char dict path:{}, label dict path: {}".format( \
            model_path, word_dict_path, head_dict_path, char_dict_path, label_dict_path))

        with Timer('Data loading'):
            print ('Task: {}'.format(task))
            allow_new_words = True
            print ('Allow new words in test data: {}'.format(allow_new_words))
            # Load word and tag dictionary
            word_dict = Dictionary(padding_token=PADDING_TOKEN, unknown_token=UNKNOWN_TOKEN)  # word tokens to Dict
            head_dict = Dictionary(padding_token=PADDING_TOKEN, unknown_token=UNKNOWN_TOKEN)  # word tokens to Dict
            char_dict = Dictionary(padding_token=PADDING_TOKEN, unknown_token=UNKNOWN_TOKEN)  # word tokens to Dict
            label_dict = Dictionary()
            dep_label_dict = Dictionary()
            word_dict.load(word_dict_path)
            head_dict.load(head_dict_path)
            char_dict.load(char_dict_path)
            label_dict.load(label_dict_path)
            dep_label_dict.load(dep_label_path)
            char_dict.accept_new, label_dict.accept_new = False, False

            data = TaggerData(config, [], [], [], [], word_dict, head_dict, char_dict, label_dict, dep_label_dict,
                              None, None)
            # Load test data.
            if task == 'srl':
                test_sentences, emb, emb_shapes = reader.get_srl_test_data(
                    args.input, config, data.word_dict, data.head_dict, data.char_dict, data.label_dict,
                    allow_new_words)
                eval_data = load_eval_data(args.input)

            print ('Read {} sentences.'.format(len(test_sentences)))
            # Add pre-trained embeddings for new words in the test data.
            # if allow_new_words:
            data.word_embeddings = emb[0]
            data.head_embeddings = emb[1]
            data.word_embedding_shapes = emb_shapes[0]
            data.head_embedding_shapes = emb_shapes[1]
            # Batching.
            test_data = data.get_test_data(test_sentences, batch_size=config.dev_batch_size)

        with Timer('Model building and loading'):
            model = BiLSTMTaggerModel(data, config=config, gpu_id=args.gpu)
            model.load(model_path)
            for param in model.parameters():
                print param.size()
            if args.gpu:
                print("Initialize the model with GPU!")
                model = model.cuda()

        with Timer('Running model'):
            dev_loss = 0.0
            srl_predictions = []

            # with torch.no_grad():  # Eval don't need the grad
            model.eval()
            for i, batched_tensor in enumerate(test_data):
                sent_ids, sent_lengths, \
                word_indexes, head_indexes, char_indexes, \
                predicate_indexes, arg_starts, arg_ends, arg_labels, srl_lens, \
                gold_predicates, num_gold_predicates = batched_tensor

                if args.gpu:
                    word_indexes, head_indexes, char_indexes, \
                    predicate_indexes, arg_starts, arg_ends, arg_labels, srl_lens = \
                        word_indexes.cuda(), head_indexes.cuda(), char_indexes.cuda(), predicate_indexes.cuda(), arg_starts.cuda(), \
                        arg_ends.cuda(), arg_labels.cuda(), srl_lens.cuda()  # , gold_predicates.cuda(), num_gold_predicates.cuda()

                predicated_dict, loss = model.forward(sent_lengths, word_indexes, head_indexes, char_indexes,
                                                      (predicate_indexes, arg_starts, arg_ends, arg_labels, srl_lens),
                                                      (gold_predicates, num_gold_predicates))
                dev_loss += float(loss)

                decoded_predictions = srl_decode(sent_lengths, predicated_dict, label_dict.idx2str, config)

                if "srl" in decoded_predictions:
                    srl_predictions.extend(decoded_predictions["srl"])

            print ('Dev loss={:.6f}'.format(dev_loss / len(test_data)))

            sentences, gold_srl, gold_ner = zip(*eval_data)
            # Summarize results, evaluate entire dev set.
            precision, recall, f1, conll_precision, conll_recall, conll_f1, ul_prec, ul_recall, ul_f1, srl_label_mat, comp = \
                (compute_srl_f1(sentences, gold_srl, srl_predictions, args.gold))

