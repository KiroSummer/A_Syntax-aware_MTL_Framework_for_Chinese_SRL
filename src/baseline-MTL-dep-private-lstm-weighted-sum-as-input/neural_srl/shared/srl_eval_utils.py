# Evaluation util functions for PropBank SRL.

import codecs
import operator
import os
import subprocess

from collections import Counter

_SRL_CONLL_EVAL_SCRIPT = "../run_eval.sh"


def split_example_for_eval(example):
    """Split document-based samples into sentence-based samples for evaluation.
      Args:
        example:
      Returns:
        Tuple of (sentence, list of SRL relations)
    """
    sentences = example["sentences"]
    num_words = sum(len(s) for s in sentences)
    word_offset = 0
    samples = []
    assert len(sentences) == 1
    for i, sentence in enumerate(sentences):
        assert i == 0  # For CoNLL-2005, there are always document == sentence.
        srl_rels = {}
        ner_spans = []  # Unused.
        for r in example["srl"][i]:
            pred_id = r[0] - word_offset
            if pred_id not in srl_rels:
                srl_rels[pred_id] = []
            srl_rels[pred_id].append((r[1] - word_offset, r[2] - word_offset, r[3]))
        samples.append((sentence, srl_rels, ner_spans))
        word_offset += len(sentence)
    return samples


def evaluate_retrieval(span_starts, span_ends, span_scores, pred_starts, pred_ends, gold_spans,
                       text_length, evaluators, debugging=False):
    """
  Evaluation for unlabeled retrieval.

  Args:
    gold_spans: Set of tuples of (start, end).
  """
    if len(span_starts) > 0:
        sorted_starts, sorted_ends, sorted_scores = zip(*sorted(
            zip(span_starts, span_ends, span_scores),
            key=operator.itemgetter(2), reverse=True))
    else:
        sorted_starts = []
        sorted_ends = []
    for k, evaluator in evaluators.items():
        if k == -3:
            predicted_spans = set(zip(span_starts, span_ends)) & gold_spans
        else:
            if k == -2:
                predicted_starts = pred_starts
                predicted_ends = pred_ends
                if debugging:
                    print "Predicted", zip(sorted_starts, sorted_ends, sorted_scores)[:len(gold_spans)]
                    print "Gold", gold_spans
            # FIXME: scalar index error
            elif k == 0:
                is_predicted = span_scores > 0
                predicted_starts = span_starts[is_predicted]
                predicted_ends = span_ends[is_predicted]
            else:
                if k == -1:
                    num_predictions = len(gold_spans)
                else:
                    num_predictions = (k * text_length) / 100
                predicted_starts = sorted_starts[:num_predictions]
                predicted_ends = sorted_ends[:num_predictions]
            predicted_spans = set(zip(predicted_starts, predicted_ends))
        evaluator.update(gold_set=gold_spans, predicted_set=predicted_spans)


def _print_f1(total_gold, total_predicted, total_matched, message=""):
    precision = 100.0 * total_matched / total_predicted if total_predicted > 0 else 0
    recall = 100.0 * total_matched / total_gold if total_gold > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    print ("{}: Precision: {}, Recall: {}, F1: {}".format(message, precision, recall, f1))
    return precision, recall, f1


def compute_span_f1(gold_data, predictions, task_name):
    assert len(gold_data) == len(predictions)
    total_gold = 0
    total_predicted = 0
    total_matched = 0
    total_unlabeled_matched = 0
    label_confusions = Counter()  # Counter of (gold, pred) label pairs.

    for i in range(len(gold_data)):
        gold = gold_data[i]
        pred = predictions[i]
        total_gold += len(gold)
        total_predicted += len(pred)
        for a0 in gold:
            for a1 in pred:
                if a0[0] == a1[0] and a0[1] == a1[1]:
                    total_unlabeled_matched += 1
                    label_confusions.update([(a0[2], a1[2]), ])
                    if a0[2] == a1[2]:
                        total_matched += 1
    prec, recall, f1 = _print_f1(total_gold, total_predicted, total_matched, task_name)
    ul_prec, ul_recall, ul_f1 = _print_f1(total_gold, total_predicted, total_unlabeled_matched,
                                          "Unlabeled " + task_name)
    return prec, recall, f1, ul_prec, ul_recall, ul_f1, label_confusions


def compute_unlabeled_span_f1(gold_data, predictions, task_name):
    assert len(gold_data) == len(predictions)
    total_gold = 0
    total_predicted = 0
    total_matched = 0
    total_unlabeled_matched = 0
    label_confusions = Counter()  # Counter of (gold, pred) label pairs.

    for i in range(len(gold_data)):
        gold = gold_data[i]
        pred = predictions[i]
        total_gold += len(gold)
        total_predicted += len(pred)
        for a0 in gold:
            for a1 in pred:
                if a0[0] == a1[0] and a0[1] == a1[1]:
                    total_unlabeled_matched += 1
                    label_confusions.update([(a0[2], a1[2]), ])
                    if a0[2] == a1[2]:
                        total_matched += 1
    prec, recall, f1 = _print_f1(total_gold, total_predicted, total_matched, task_name)
    ul_prec, ul_recall, ul_f1 = _print_f1(total_gold, total_predicted, total_unlabeled_matched,
                                          "Unlabeled " + task_name)
    return prec, recall, f1, ul_prec, ul_recall, ul_f1, label_confusions


def compute_srl_f1(sentences, gold_srl, predictions, srl_conll_eval_path):
    assert len(gold_srl) == len(predictions)
    total_gold = 0
    total_predicted = 0
    total_matched = 0
    total_unlabeled_matched = 0
    comp_sents = 0
    label_confusions = Counter()

    # Compute unofficial F1 of SRL relations.
    for gold, prediction in zip(gold_srl, predictions):
        gold_rels = 0
        pred_rels = 0
        matched = 0
        for pred_id, gold_args in gold.iteritems():
            filtered_gold_args = [a for a in gold_args if a[2] not in ["V", "C-V"]]
            total_gold += len(filtered_gold_args)
            gold_rels += len(filtered_gold_args)
            if pred_id not in prediction:
                continue
            for a0 in filtered_gold_args:
                for a1 in prediction[pred_id]:
                    if a0[0] == a1[0] and a0[1] == a1[1]:
                        total_unlabeled_matched += 1
                        label_confusions.update([(a0[2], a1[2]), ])
                        if a0[2] == a1[2]:
                            total_matched += 1
                            matched += 1
        for pred_id, args in prediction.iteritems():
            filtered_args = [a for a in args if a[2] not in ["V"]]  # "C-V"]]
            total_predicted += len(filtered_args)
            pred_rels += len(filtered_args)

        if gold_rels == matched and pred_rels == matched:
            comp_sents += 1

    precision, recall, f1 = _print_f1(total_gold, total_predicted, total_matched, "SRL (unofficial)")
    ul_prec, ul_recall, ul_f1 = _print_f1(total_gold, total_predicted, total_unlabeled_matched,
                                          "Unlabeled SRL (unofficial)")

    # Prepare to compute official F1.
    if not srl_conll_eval_path:
        print "No gold conll_eval data provided. Recreating ..."
        gold_path = "/tmp/srl_pred_%d.gold" % os.getpid()
        print_to_conll(sentences, gold_srl, gold_path, None)
        gold_predicates = None
    else:
        gold_path = srl_conll_eval_path
        gold_predicates = read_gold_predicates(gold_path)

    temp_output = "/tmp/srl_pred_%d.tmp" % os.getpid()
    print("Output temp outoput {}".format(temp_output))
    print_to_conll(sentences, predictions, temp_output, gold_predicates)

    # Evalute twice with official script.
    child = subprocess.Popen('sh {} {} {}'.format(
        _SRL_CONLL_EVAL_SCRIPT, gold_path, temp_output), shell=True, stdout=subprocess.PIPE)
    eval_info = child.communicate()[0]
    child2 = subprocess.Popen('sh {} {} {}'.format(
        _SRL_CONLL_EVAL_SCRIPT, temp_output, gold_path), shell=True, stdout=subprocess.PIPE)
    eval_info2 = child2.communicate()[0]
    try:
        conll_recall = float(eval_info.strip().split("\n")[6].strip().split()[5])
        conll_precision = float(eval_info2.strip().split("\n")[6].strip().split()[5])
        if conll_recall + conll_precision > 0:
            conll_f1 = 2 * conll_recall * conll_precision / (conll_recall + conll_precision)
        else:
            conll_f1 = 0
        print(eval_info)
        print(eval_info2)
        print("Official CoNLL Precision={}, Recall={}, Fscore={}".format(
            conll_precision, conll_recall, conll_f1))
    except IndexError:
        conll_recall = 0
        conll_precision = 0
        conll_f1 = 0
        print("Unable to get FScore. Skipping.")
    return precision, recall, f1, conll_precision, conll_recall, conll_f1, ul_prec, ul_recall, ul_f1, label_confusions, comp_sents


def print_sentence_to_conll(fout, tokens, labels):
    """Print a labeled sentence into CoNLL format.
  """
    for label_column in labels:
        assert len(label_column) == len(tokens)
    for i in range(len(tokens)):
        fout.write(tokens[i].ljust(15))
        for label_column in labels:
            fout.write(label_column[i].rjust(15))
        fout.write("\n")
    fout.write("\n")


def read_gold_predicates(gold_path):
    print "gold path", gold_path
    fin = codecs.open(gold_path, "r", "utf-8")
    gold_predicates = [[], ]
    for line in fin:
        line = line.strip()
        if not line:
            gold_predicates.append([])
        else:
            info = line.split()
            gold_predicates[-1].append(info[0])
    fin.close()
    return gold_predicates


def print_to_conll(sentences, srl_labels, output_filename, gold_predicates):
    fout = codecs.open(output_filename, "w", "utf-8")
    for sent_id, words in enumerate(sentences):
        if gold_predicates:
            assert len(gold_predicates[sent_id]) == len(words)
        pred_to_args = srl_labels[sent_id]
        props = ["-" for _ in words]
        col_labels = [["*" for _ in words] for _ in range(len(pred_to_args))]
        for i, pred_id in enumerate(sorted(pred_to_args.keys())):
            # To make sure CoNLL-eval script count matching predicates as correct.
            if gold_predicates and gold_predicates[sent_id][pred_id] != "-":
                props[pred_id] = gold_predicates[sent_id][pred_id]
            else:
                props[pred_id] = "P" + words[pred_id]
            flags = [False for _ in words]
            for start, end, label in pred_to_args[pred_id]:
                if not max(flags[start:end + 1]):
                    col_labels[i][start] = "(" + label + col_labels[i][start]
                    col_labels[i][end] = col_labels[i][end] + ")"
                    for j in range(start, end + 1):
                        flags[j] = True
            # Add unpredicted verb (for predicted SRL).
            if not flags[pred_id]:  # if the predicate id is False
                col_labels[i][pred_id] = "(V*)"
        print_sentence_to_conll(fout, props, col_labels)
    fout.close()
