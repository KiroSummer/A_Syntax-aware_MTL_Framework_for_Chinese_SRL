import numpy as np
import codecs
from dictionary import *
from constants import *


class SyntacticTree(object):
    def __init__(self, sentence_id):
        self.sentence_id = sentence_id
        self.word_forms = ["Root"]
        self.word_forms_ids = []
        self.char_ids = [[]]  # 2D
        self.pos_forms = ["Root"]
        self.heads = [0]
        self.labels = ["Root"]
        self.labels_id = []


class SyntacticCONLL(object):
    def __init__(self):
        self.file_name = ""
        self.trees = []
        self.sample_dep_data = None

    def read_from_file(self, filename, max_sentence_length=100, prune_ratio=0.8):
        self.file_name = filename

        print("Reading conll syntactic trees from {} and the prune ratio is {}".format(self.file_name, prune_ratio))
        conll_file = codecs.open(self.file_name, 'r', encoding="utf8")
        if conll_file.closed:
            print("Cannot open the syntactic conll file! Please check {}".format(self.file_name))

        sentence_id = 0
        a_tree = SyntacticTree(sentence_id)
        for line in conll_file:
            if line == '\n' or line == '\r\n':  # new sentence
                sentence_id += 1
                if len(a_tree.word_forms) <= max_sentence_length:
                    # keep the sentence with the length < max_sentence_l
                    self.trees.append(a_tree)
                a_tree = SyntacticTree(sentence_id)
                continue
            tokens = line.strip().split('\t')
            a_tree.word_forms.append(tokens[1])
            a_tree.pos_forms.append(tokens[3])
            # head = int(tokens[6]) if int(tokens[6]) > 0 else -1
            head = int(tokens[6])  # root's head is 0
            a_tree.heads.append(head)
            a_tree.labels.append(tokens[7])
            dep_prob = float(tokens[9])
            if dep_prob < prune_ratio:
                a_tree.heads[-1] = -1
        print("Total {} conll trees, load {} conll syntactic trees.".format(sentence_id, len(self.trees)))

    @staticmethod
    def list_of_words_to_ids(list_of_words, dictionary, lowercase=False, pretrained_embeddings=None):
        ids = []
        for s in list_of_words:
            s = s.encode("utf8")
            if s is None:
                ids.append(-1)
                continue
            if lowercase:
                s = s.lower()
            if (pretrained_embeddings is not None) and (s not in pretrained_embeddings):
                s = UNKNOWN_TOKEN
            ids.append(dictionary.add(s))
        return ids

    def tokenize_dep_trees(self, word_dict, char_dict, syn_label_dict, pretrained_word_embedding=None):
        for tree in self.trees:
            tree.word_forms_ids = SyntacticCONLL.list_of_words_to_ids(tree.word_forms, word_dict, False,
                                                                      pretrained_word_embedding)
            words = tree.word_forms
            max_word_length = max([len(w) for w in words] + [3, 4, 5])  # compare with character cnn filter width
            single_sample_char_tokens = np.zeros([len(words), max_word_length], dtype=np.int64)
            for i, word in enumerate(words):
                single_sample_char_tokens[i, :len(word)] = SyntacticCONLL.list_of_words_to_ids(word, char_dict)
            # Add the sample char tokens into the sample_char_tokens
            tree.char_ids = single_sample_char_tokens

            tree.labels_id = SyntacticCONLL.list_of_words_to_ids(tree.labels, syn_label_dict)

        sample_word_forms_ids = [tree.word_forms_ids for tree in self.trees]
        sample_char_ids = [tree.char_ids for tree in self.trees]
        sample_heads = [np.asarray(tree.heads) for tree in self.trees]
        sample_labels_ids = [np.asarray(tree.labels_id) for tree in self.trees]
        self.sample_dep_data = zip(sample_word_forms_ids, sample_char_ids, sample_heads, sample_labels_ids)

    def get_syntactic_label_dict(self, syn_label_dict=None):
        if syn_label_dict is None:
            syn_label_dict = Dictionary(padding_token=PADDING_TOKEN, unknown_token=UNKNOWN_TOKEN)
        else:
            assert syn_label_dict.accept_new == False
        sentences_length = len(self.trees)
        for i in range(sentences_length):
            ith_sentence_length = len(self.trees[i].labels)
            for j in range(ith_sentence_length):
                self.trees[i].labels_id.append(syn_label_dict.add(self.trees[i].labels[j]))
        return syn_label_dict


class SyntacticRepresentation(object):
    def __init__(self):
        self.file_name = ""
        self.representations = []

    def read_from_file(self, filename):
        self.file_name = filename
        print("Reading lstm representations from {}".format(self.file_name))
        representation_file = open(self.file_name, 'r')
        if representation_file.closed:
            print("Cannot open the representation file! Please check {}".format(self.file_name))
            exit()
        each_sentence_representations = []
        for line in representation_file:
            if line == '\n' or line == "\r\n":  # new sentence
                self.representations.append(each_sentence_representations)
                each_sentence_representations = []
                continue
            line = line.strip()
            line = line.split('\t')
            line = line[1].split(' ')
            rep = np.asarray(line, dtype=np.float32)
            each_sentence_representations.append(rep)
        representation_file.close()
        print("Load LSTM representations done, total {} sentences' representations".format(len(self.representations)))

    def minus_by_the_predicate(self, corpus_tensors):
        has_processed_sentence_id = {}
        for i, data in enumerate(corpus_tensors):
            sentence_id = data[0][0][0]
            predicates = data[0][2]
            predicate_id = predicates.argmax()
            if sentence_id in has_processed_sentence_id:
                continue
            else:
                has_processed_sentence_id[sentence_id] = 1
            for j in range(1, len(self.representations[sentence_id])):  # Root doesn't use.
                self.representations[sentence_id][j] = self.representations[sentence_id][predicate_id] \
                                                       - self.representations[sentence_id][j]

    def check_math_corpus(self, lengths):
        for i, length in enumerate(lengths):
            if len(self.representations[i]) != length + 1:  # 1 means the first one, Root. Actually never use it.
                print i, length, len(self.representations[i])
                print("sentence {} doesn't match: lstm representation {} vs corpus {}" \
                      .format(i, len(self.representations[i])), length)
                exit()
        print("LSTM representation match the corpus!")
