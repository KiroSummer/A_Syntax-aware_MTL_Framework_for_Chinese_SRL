import json
import codecs
import numpy as np

from constants import *
from dictionary import Dictionary
from srl_eval_utils import *
from syntactic_extraction import *
from measurements import Timer


def get_sentences(filepath, use_se_marker=False):
    """ Read tokenized sentences from file """
    sentences = []
    with open(filepath) as f:
        for line in f.readlines():
            inputs = line.strip().split('|||')
            lefthand_input = inputs[0].strip().split()
            # If gold tags are not provided, create a sequence of dummy tags.
            righthand_input = inputs[1].strip().split() if len(inputs) > 1 \
                else ['O' for _ in lefthand_input]
            if use_se_marker:
                words = [START_MARKER] + lefthand_input + [END_MARKER]
                labels = [None] + righthand_input + [None]
            else:
                words = lefthand_input
                labels = righthand_input
            sentences.append((words, labels))
    return sentences


class srl_sentence():
    def __init__(self, obj):
        self.speakers = obj["speakers"]
        self.doc_key = obj["doc_key"]
        self.sentences = obj["sentences"]
        self.srl = obj["srl"][0]
        self.constituents = obj["constituents"]
        self.clusters = obj["clusters"]

    def get_labels(self, dictionary):
        ids = []
        for s in self.srl:
            s = s[-1]
            if s is None:
                ids.append(-1)
                continue
            ids.append(dictionary.add(s))
        return ids

    def tokenize_argument_spans(self, dictionary):
        srl_span = []
        for srl_label in self.srl:  # remove self-loop V-V
            if srl_label[-1] in ["V", "C-V"]:
                continue
                dictionary.add(srl_label[-1])
                continue
            srl_span.append([int(srl_label[0]),
                             int(srl_label[1]), int(srl_label[2]), int(dictionary.add(srl_label[3]))])
        if len(srl_span) == 0:  # if the sentence has no arguments.
            return [[], [], [], []]
        tokenized_predicates, tokenized_arg_starts, tokenized_arg_ends, tokenized_arg_labels = zip(*srl_span)
        return tokenized_predicates, tokenized_arg_starts, tokenized_arg_ends, tokenized_arg_labels

    def get_all_gold_predicates(self):
        if len(self.srl) > 0:
            predicates, _, _, _ = zip(*self.srl)
        else:
            predicates = []
        predicates = np.unique(predicates)
        return predicates, len(predicates)


def get_srl_sentences(filepath):
    """
    Data loading with json format.
    """
    sentences = []
    with codecs.open(filepath, encoding="utf8") as f:
        for line in f.readlines():
            sen = json.loads(line)
            srl_sen = srl_sentence(sen)
            sentences.append(srl_sen)
        print("{} total sentences number {}".format(filepath, len(sentences)))
    return sentences


def normalize(v):
    norm = np.linalg.norm(v)
    if norm > 0:
        return v / norm
    else:
        return v


def get_pretrained_embeddings(filepath):
    embeddings = dict()
    with open(filepath, 'r') as f:
        for line in f:
            info = line.strip().split()
            embeddings[info[0]] = normalize(np.asarray([float(r) for r in info[1:]]))  # normalize the embedding
        f.close()
    embedding_size = len(embeddings.values()[0])
    print '{}, embedding size={}, embedding number={}'.format(filepath, embedding_size, len(embeddings))
    # add START and END MARKER, PADDING and UNKNOWN token in embedding dict
    embeddings[START_MARKER] = np.asarray([random.gauss(0, 0.01) for _ in range(embedding_size)])
    embeddings[END_MARKER] = np.asarray([random.gauss(0, 0.01) for _ in range(embedding_size)])
    # set padding the unknown token id into the embeddings
    if PADDING_TOKEN not in embeddings:
        embeddings[PADDING_TOKEN] = np.zeros(embedding_size)
    if UNKNOWN_TOKEN not in embeddings:
        embeddings[UNKNOWN_TOKEN] = np.zeros(embedding_size)
    return embeddings


def tokenize_data(data, word_dict, head_dict, char_dict, label_dict, lowercase=False, pretrained_word_embedding=None,
                  pretrained_head_embedding=None):
    """
    :param data: the raw input sentences
    :param word_dict: word dictionary
    :param head_dict: label dictionary
    :param char_dict: character dictionary
    :param label_dict: srl label dictionary
    :param lowercase: bool value, if the word or character needs to lower
    :param pretrained_word_embedding: pre-trained word embedding
    :param pretrained_head_embedding: pre-trained head embedding
    :return: a list storing the [sentence id, length, [words], [heads], [characters], [srl argument spans]]
    """
    sample_ids = [int(sent.doc_key.encode('utf-8')[1:]) for sent in data]  # convert the doc_id to int
    sample_word_tokens = [list_of_words_to_ids(sent.sentences[0], word_dict, lowercase, pretrained_word_embedding)
                          for sent in data]  # sent.sentences[0] is the words of the sentence
    sample_head_tokens = [list_of_words_to_ids(sent.sentences[0], head_dict, lowercase, pretrained_head_embedding)
                          for sent in data]
    # for the character
    sample_char_tokens = []
    for sent in data:
        words = sent.sentences[0]
        max_word_length = max([len(w) for w in words] + [3, 4, 5])  # compare with character cnn filter width
        single_sample_char_tokens = np.zeros([len(words), max_word_length], dtype=np.int64)
        for i, word in enumerate(words):
            single_sample_char_tokens[i, :len(word)] = list_of_words_to_ids(word, char_dict, lowercase)
        # Add the sample char tokens into the sample_char_tokens
        sample_char_tokens.append(single_sample_char_tokens)
    sample_lengths = [len(sent.sentences[0])for sent in data]
    sample_arg_span_tokens = [sent.tokenize_argument_spans(label_dict) for sent in data]
    sample_gold_predicates = [sent.get_all_gold_predicates() for sent in data]
    return zip(sample_ids, sample_lengths, sample_word_tokens, sample_head_tokens, sample_char_tokens,
               sample_arg_span_tokens, sample_gold_predicates)


def list_of_words_to_ids(list_of_words, dictionary, lowercase=False, pretrained_embeddings=None):
    ids = []
    for s in list_of_words:
        s = s.encode('utf-8')  # unicode -> utf-8
        if s is None:
            ids.append(-1)
            continue
        if lowercase:
            s = s.lower()
        if (pretrained_embeddings is not None) and (s not in pretrained_embeddings):
            s = UNKNOWN_TOKEN
        ids.append(dictionary.add(s))
    return ids


def load_eval_data(eval_path):
    eval_data = []
    with open(eval_path, 'r') as f:
        eval_examples = [json.loads(jsonline) for jsonline in f.readlines()]
    for doc_id, example in enumerate(eval_examples):
        eval_data.extend(split_example_for_eval(example))
    print("Loaded {} eval examples.".format(len(eval_data)))
    return eval_data


def get_srl_data(config, train_data_path, dep_path, dev_data_path, vocab_path=None, char_path=None, label_path=None):
    # Load sentences (documents) from data paths respectively.
    raw_train_sents = get_srl_sentences(train_data_path)
    raw_dev_sents = get_srl_sentences(dev_data_path)
    # Load dev data
    eval_data = load_eval_data(dev_data_path)
    # Load pretrained embeddings
    word_embeddings = get_pretrained_embeddings(config.word_embedding)  # get pre-trained embeddings
    head_embeddings = get_pretrained_embeddings(config.head_embedding)

    # Prepare word embedding dictionary.
    word_dict = Dictionary(padding_token=PADDING_TOKEN, unknown_token=UNKNOWN_TOKEN)
    # Prepare head embedding dictionary.
    head_dict = Dictionary(padding_token=PADDING_TOKEN, unknown_token=UNKNOWN_TOKEN)
    # Prepare char dictionary.
    char_dict = Dictionary(padding_token=PADDING_TOKEN, unknown_token=UNKNOWN_TOKEN)
    with open(char_path, 'r') as f_char:
        for line in f_char:
            char_dict.add(line.strip())
        f_char.close()
    char_dict.accept_new = False
    print 'Load {} chars, Dictionary freezed.'.format(char_dict.size())
    # Parpare SRL label dictionary.
    label_dict = Dictionary()
    label_dict.set_unknown_token(NULL_LABEL)  # train corpus contains the label 'O' ?
    if label_path is not None:
        with open(label_path, 'r') as f_labels:
            for line in f_labels:
                label_dict.add(line.strip())
            f_labels.close()
        label_dict.set_unknown_token(NULL_LABEL)
        label_dict.accept_new = False
        print 'Load {} labels. Dictionary freezed.'.format(label_dict.size())
    # Parpare SRL label dictionary.
    dep_label_dict = Dictionary()

    # Training data: Get tokens and labels: [sentence_id, word, predicate, label]
    train_samples = tokenize_data(raw_train_sents, word_dict, head_dict, char_dict, label_dict, False,
                                  word_embeddings, head_embeddings)
    # Data for dep Trees
    with Timer("Loading Dependency Trees"):
        dep_trees = SyntacticCONLL()
        dep_trees.read_from_file(dep_path, prune_ratio=config.dep_prune_ratio)
        dep_trees.tokenize_dep_trees(word_dict, char_dict, dep_label_dict, word_embeddings)

    # set dictionary freezed
    char_dict.accept_new, label_dict.accept_new, dep_label_dict.accept_new = False, False, False
    # Development data:
    dev_samples = tokenize_data(raw_dev_sents, word_dict, head_dict, char_dict, label_dict, False,
                                word_embeddings, head_embeddings)

    # set word and head dict freezed.
    word_dict.accept_new, head_dict.accept_new = False, False

    print("Extract {} words and {} tags".format(word_dict.size(), label_dict.size()))
    print("Max training sentence length: {}".format(max([s[1] for s in train_samples])))
    print("Max development sentence length: {}".format(max([s[1] for s in dev_samples])))

    word_embedding = np.asarray([word_embeddings[w] for w in word_dict.idx2str])
    word_embedding_shape = [len(word_embedding), len(word_embedding[0])]
    head_embedding = np.asarray([head_embeddings[w] for w in head_dict.idx2str])
    head_embedding_shape = [len(head_embedding), len(head_embedding[0])]
    print("word embedding shape {}, head embedding shape {}".format(word_embedding_shape, head_embedding_shape))
    return (train_samples, dev_samples, dep_trees.sample_dep_data, eval_data,
            word_dict, head_dict, char_dict, label_dict, dep_label_dict,
            [word_embedding, head_embedding],
            [word_embedding_shape, head_embedding_shape])


def get_srl_test_data(filepath, config, word_dict, head_dict, char_dict, label_dict, allow_new_words=True):
    """get the test data from file"""
    word_dict.accept_new = allow_new_words
    if label_dict.accept_new:
        label_dict.set_unknown_token(NULL_LABEL)
        label_dict.accept_new = False

    if filepath is not None and filepath != '':
        samples = get_srl_sentences(filepath)
    else:
        samples = []
    word_to_embeddings = get_pretrained_embeddings(config.word_embedding)
    head_to_embeddings = get_pretrained_embeddings(config.head_embedding)
    test_samples = []
    if allow_new_words:
        test_samples = tokenize_data(samples, word_dict, head_dict, char_dict, label_dict, False,
                                     word_to_embeddings, head_to_embeddings)
        # tokens = [list_of_words_to_ids(sent[1], word_dict, True, word_to_embeddings) for sent in samples]
    else:
        tokens = [list_of_words_to_ids(sent[1], word_dict, True) for sent in samples]

    word_embedding = np.asarray([word_to_embeddings[w] for w in word_dict.idx2str])
    word_embedding_shape = [len(word_embedding), len(word_embedding[0])]
    head_embedding = np.asarray([head_to_embeddings[w] for w in head_dict.idx2str])
    head_embedding_shape = [len(head_embedding), len(head_embedding[0])]
    return (test_samples, [word_embedding, head_embedding], [word_embedding_shape, head_embedding_shape])
