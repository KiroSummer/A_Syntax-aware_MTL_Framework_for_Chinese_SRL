''' Full suite of error analysis on CoNLL05 '''

import itertools
from itertools import izip
import sys
import re

# from wsj_syntax_helper import extract_gold_syntax_spans

CORE_ROLES = {'A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'AA' }

MAX_LEN = 200
#CONLL05_GOLD_SYNTAX = 'conll05.devel.gold.syntax'
#CONLL05_GOLD_SRL = 'conll05.test.wsj.props.gold.txt'
CONLL05_GOLD_SRL = 'cpb.1.0.test.props'
#CONLL05_GOLD_SRL = 'conll05.test.brown.props.gold.txt'
#CONLL05_GOLD_SRL= 'conll12.test.gold'

def read_file(filename):
  '''
  '''
  fin = open(filename, 'r')
  sentences = []
  gold = []
  predicted = []

  s0 = []
  g0 = []
  p0 = []

  for line in fin:
    line = line.strip()
    if line == '':
      sentences.append(s0)
      gold.append(g0)
      predicted.append(p0)
      s0 = []
      g0 = []
      p0 = []
      continue

    info = line.split()
    if len(info) == 2:
      continue
    s0.append(info[0])
    g0.append(info[1])
    p0.append(info[-2])

  fin.close()
  return sentences, gold, predicted

def read_conll_prediction(filename):
  '''
  '''
  fin = open(filename, 'r')
  sentences = []
  predicates = []
  arguments = []

  s0 = []
  p0 = []
  a0 = []

  for line in fin:
    line = line.strip()
    if line == '' and len(s0) > 0:
      sentences.append(s0)
      predicates.append(p0)
      arguments.append(a0)
      s0 = []
      p0 = []
      a0 = []
      continue

    info = line.split()
    s0.append(info[0])
    index = len(s0) - 1
    if index == 0:
      tags = [[] for _ in info[1:]]
      spans = ["" for _ in info[1:]]
      a0 = [[] for _ in info[1:]]

    is_predicate = (info[0] != '-')
    for t in range(len(tags)):
      arg = info[1 + t]
      label = arg.strip("()*)").split("*")[0]

      if "(" in arg:
        tags[t].append("B-" + label)
        spans[t] = label
        a0[t].append([label, index, -1])
      elif spans[t] != "":
        tags[t].append("I-" + spans[t])
      else:
        tags[t].append("O")
      if ")" in arg:
        spans[t] = ""
        a0[t][-1][2] = index

    if is_predicate:
      p0.append(index)

  if len(s0) > 0:
    sentences.append(s0)
    predicates.append(p0)
    arguments.append(a0)

  fin.close()
  return sentences, predicates, arguments 


def extract_spans(tags):
  spans = []
  curr = ['', -1, -1]
  for i in range(len(tags)):
    tag = tags[i]
    if curr[0] != '' and not (tag[0] == 'I' and tag[2:] == curr[0]):
      curr[2] = i
      spans.append([curr[0], curr[1], curr[2]])
      curr = ['', -1, -1]

    if tag[0] == 'B' or (tag[0] == 'I' and tag[2:] != curr[0]):
      curr[0] = tag[2:]
      curr[1] = i

  if curr[0] != '':
    curr[2] = len(tags)
    spans.append([curr[0], curr[1], curr[2]])

  #print tags
  #print '\n'.join([str(s) for s in spans])

  return spans

def find(trg, spans):
  for s in spans:
    if trg[0] == s[0] and trg[1] == s[1] and trg[2] == s[2]:
      return True
  return False

def unlabeled_find(trg, spans):
  for s in spans:
    if trg[1] == s[1] and trg[2] == s[2]:
      return True
  return False

def is_base_role(label):
  return label[0] != 'C' and label[0] != 'R'

def get_base_role(label):
  return label.split('C-')[-1].split('R-')[-1]

def compute_srl_violations(spans):
  ''' Unique core violations, continuation, reference
  '''
  uv = [0, 0]
  cv = [0, 0]
  rv = [0, 0]

  seen_roles = set([])
  all_roles = set([s[0] for s in spans if is_base_role(s[0])])

  for i, span in enumerate(spans):
    label = span[0]
    if label[0] == 'C':
      cv[1] += 1
      if not (label[2:] in seen_roles):
        cv[0] += 1
    elif label[0] == 'R':
      rv[1] += 1
      if not (label[2:] in all_roles):
        rv[0] += 1
    elif label in CORE_ROLES:
      uv[1] += 1
      if label in seen_roles:
        uv[0] += 1

    if is_base_role(label):
      seen_roles.add(label)
  
  return uv, cv, rv  

def compute_syntax_match(srl_spans, syn_spans):
  ''' Return: number of srl spans matching syntactic spans.
  '''
  num_matched, num_processed = 0, 0
  for s in srl_spans:
    #if s[2] > s[1]: # skip singletons
    num_processed += 1
    if s[1] == s[2] or unlabeled_find(s, syn_spans):
      num_matched += 1
      continue
    
    # Node concatenation
    #matched = False
    #for a in syn_spans:
    #  for b in syn_spans:
    #    if a[2] + 1 == b[1] and s[1] == a[1] and s[2] == b[2]:
    #      num_matched += 1
    #      matched = True
    #      break
    #  if matched: break
    #else:
    #if not matched:
    #  print s, '\n', syn_spans, '\n'

  return num_matched, num_processed

def get_surface_dist(prop_id, span):
  return min(abs(prop_id - span[1]), abs(prop_id - span[2]))

def get_num_intv_args(prop_id, span, spans):
  ''' Number of intervening arguments.
  '''
  num_intv = 0
  for s in spans:
    if (prop_id < s[1] and s[2] < span[1])  \
        or (span[2] < s[1] and s[2] < prop_id):
      num_intv += 1
  return num_intv


def new_histogram(num_bins, num_stats):
  hist = []
  for i in range(num_bins):
    hist.append([0] * num_stats)
  return hist

def add_to_histogram(hist, new_stats):
  for h, stats in izip(hist, new_stats):
    for i, s in enumerate(stats):
      h[i] += s



def breakdown_acc_by_arg_length(gold_spans, gold_match, pred_spans, pred_match):
  ''' Breakdown accuracy by argument length.
  '''
  recall = new_histogram(MAX_LEN, 2)
  precision = new_histogram(MAX_LEN, 2)

  for span, matched in izip(gold_spans, gold_match):
    slen = span[2] - span[1] + 1
    recall[slen][1] += 1
    recall[slen][0] += matched

  for span, matched in izip(pred_spans, pred_match):
    slen = span[2] - span[1] + 1
    precision[slen][1] += 1
    precision[slen][0] += matched

  return precision, recall

def breakdown_acc_by_dist_to_prop(gold_spans, gold_match, pred_spans, pred_match, prop_id):
  ''' Breakdown accuracy by distance to predicate (min).
  '''
  recall = new_histogram(MAX_LEN, 2)
  precision = new_histogram(MAX_LEN, 2)

  for span, matched in izip(gold_spans, gold_match):
    dist = get_surface_dist(prop_id, span)
    recall[dist][1] += 1
    recall[dist][0] += matched

  for span, matched in izip(pred_spans, pred_match):
    dist = get_surface_dist(prop_id, span)
    precision[dist][1] += 1
    precision[dist][0] += matched

  return precision, recall

def breakdown_acc_by_num_intv_args(gold_spans, gold_match, pred_spans, pred_match, prop_id):
  ''' Breakdown accuracy by distance to predicate (min).
  '''
  recall = new_histogram(MAX_LEN, 2)
  precision = new_histogram(MAX_LEN, 2)

  for span, matched in izip(gold_spans, gold_match):
    dist = get_num_intv_args(prop_id, span, gold_spans)
    recall[dist][1] += 1
    recall[dist][0] += matched

  for span, matched in izip(pred_spans, pred_match):
    dist = get_num_intv_args(prop_id, span, pred_spans)
    precision[dist][1] += 1
    precision[dist][0] += matched

  return precision, recall

def breakdown_acc_by_slen(gold_spans, gold_match, pred_spans, pred_match, slen):
  ''' Breakdown accuracy by sentence length.
  '''
  recall = new_histogram(MAX_LEN, 2)
  precision = new_histogram(MAX_LEN, 2)

  for span, matched in izip(gold_spans, gold_match):
    recall[slen][1] += 1
    recall[slen][0] += matched

  for span, matched in izip(pred_spans, pred_match):
    precision[slen][1] += 1
    precision[slen][0] += matched

  return precision, recall

def print_histogram(name, p_hist, r_hist, buckets):
  new_p = new_histogram(len(buckets), len(p_hist[0]))
  new_r = new_histogram(len(buckets), len(r_hist[0]))

  for i, p in enumerate(p_hist):
    for bucket_id, b in enumerate(buckets):
      if b[0] <= i and i <= b[1]:
        new_p[bucket_id][0] += p[0]
        new_p[bucket_id][1] += p[1]

  for i, r in enumerate(r_hist):
    for bucket_id, b in enumerate(buckets):
      if b[0] <= i and i <= b[1]:
        new_r[bucket_id][0] += r[0]
        new_r[bucket_id][1] += r[1]

 
  print 'Precision breakdown by {}:'.format(name)
  for i, b in enumerate(buckets):
    p = new_p[i]
    print '{}-{}\t{}\t{}'.format(b[0], b[1], p[1], 100.0 * p[0] / p[1])

  print 'Recall breakdown by {}:'.format(name)
  for i, b in enumerate(buckets):
    r = new_r[i]
    print '{}-{}\t{}\t{}'.format(b[0], b[1], r[1], 100.0 * r[0] / r[1])

  print 'F1 breakdown by {}:'.format(name)
  for i, b in enumerate(buckets):
    p = 100.0 * new_p[i][0] / new_p[i][1]
    r = 100.0 * new_r[i][0] / new_r[i][1]
    f1 = 2 * p * r / (p + r)
    print '{}-{}\t{}'.format(b[0], b[1], f1)
 

if __name__ == '__main__':
  sentences, predicates, gold = read_conll_prediction(CONLL05_GOLD_SRL)
  _, _, predicted = read_conll_prediction(sys.argv[1])
  # _, postags, syn_spans = extract_gold_syntax_spans(CONLL05_GOLD_SYNTAX)

  #print len(sentences), len(predicates), len(syn_spans), len(gold), len(predicted)
  assert len(gold) == len(predicted)

  num_matched = 0
  num_gold = 0
  num_predicted = 0

  gold_violations = [[0, 0], [0, 0], [0, 0]]
  pred_violations = [[0, 0], [0, 0], [0, 0]]

  gold_synmatch = [0, 0]
  pred_synmatch = [0, 0]

  # By argument span length
  p_arglen = new_histogram(MAX_LEN, 2)
  r_arglen = new_histogram(MAX_LEN, 2)

  # By distance to predicate
  p_propdist = new_histogram(MAX_LEN, 2)
  r_propdist = new_histogram(MAX_LEN, 2)

  # by number of intv args
  p_iargs = new_histogram(MAX_LEN, 2)
  r_iargs = new_histogram(MAX_LEN, 2)

  # By distance to sentence beginning
  p_sdist = new_histogram(MAX_LEN, 2)
  r_sdist = new_histogram(MAX_LEN, 2)

  # By sentence length.
  p_slen = new_histogram(MAX_LEN, 2)
  r_slen = new_histogram(MAX_LEN, 2)

  sentence_id = -1
  for s0, props, g0, p0 in izip(sentences, predicates, gold, predicted):
    num_gold, num_predicted, num_matched = 0, 0, 0
    # print "s0", s0, '\n', "props", props, '\n', "g0", g0, '\n', 's0', s0
    for prop_id, gold_args, pred_args in izip(props, g0, p0):
      gold_spans = [s for s in gold_args if s[0] != 'V' and not 'C-V' in s[0]]
      pred_spans = [s for s in pred_args if s[0] != 'V' and not 'C-V' in s[0]]

      # Compute F1
      gold_matched = [find(g, pred_spans) for g in gold_spans]
      pred_matched = [find(p, gold_spans) for p in pred_spans]
      # print gold_spans, '\n', pred_spans, '\n', pred_matched

      num_gold += len(gold_spans)
      num_predicted += len(pred_spans)
      num_matched += sum(pred_matched)

    if num_gold == 0:
      continue
    sentence_id += 1
    if num_predicted == 0:
      precision = 100.0 * num_matched / num_predicted
      recall = 100.0 * num_matched / num_gold
      print '\t', sentence_id, '\t', num_gold, '\t', 0, '\t', "%.2f"%0.0, '\t', "%.2f"%recall, '\t', num_matched, '\t', num_matched, '\t', num_gold, '\t', "0 0 0 0"
      continue
    precision = 100.0 * num_matched / num_predicted
    recall = 100.0 * num_matched / num_gold
    # f1 = 2.0 * precision * recall / (precision + recall)

    print '\t', sentence_id, '\t', num_gold, '\t', 0, '\t', "%.2f"%precision, '\t', "%.2f"%recall, '\t', num_matched, '\t', num_matched, '\t', num_gold, '\t', "0 0 0 0"

