from constants import UNKNOWN_TOKEN
from constants import PADDING_TOKEN

import numpy as np
import random
import torch
from torch.autograd.variable import Variable


def tensorize(sentence, max_length):
    """ Input:
      - sentence: The sentence is a tuple of lists (s1, s2, ..., sk)
            s1 is always a sequence of word ids.
            sk is always a sequence of label ids.
            s2 ... sk-1 are sequences of feature ids,
              such as predicate or supertag features.
      - max_length: The maximum length of sequences, used for padding.
  """
    x = np.array([t for t in zip(*sentence[1:3])])
    x = x.transpose()  # SRL: (T, 2) T is the sentence length
    return {"sentence_id": sentence[0], "inputs": x, "chars": sentence[3],
            "labels": sentence[-1], "sentence_length": len(sentence[0])}


class TaggerData(object):
    def __init__(self, config, train_sents, dev_sents, dep_data, eval_data, word_dict, head_dict, char_dict,
                 label_dict, dep_label_dict,
                 embeddings, embeddings_shapes):
        self.max_train_length = config.max_train_length
        self.max_dev_length = max([s[1] for s in dev_sents]) if len(dev_sents) > 0 else 0
        self.batch_size = config.batch_size
        self.max_tokens_per_batch = config.max_tokens_per_batch

        self.train_samples = [s for s in train_sents if s[1] <= self.max_train_length]
        self.dev_samples = dev_sents
        self.dep_data = dep_data
        self.eval_data = eval_data
        self.word_dict = word_dict
        self.head_dict = head_dict
        self.char_dict = char_dict
        self.label_dict = label_dict
        self.dep_label_dict = dep_label_dict
        self.word_embeddings, self.head_embeddings = None, None
        self.word_embedding_shapes, self.head_embedding_shapes = None, None

        self.word_padding_id = word_dict.str2idx[PADDING_TOKEN]
        self.word_unk_id = word_dict.str2idx[UNKNOWN_TOKEN]
        self.head_padding_id = head_dict.str2idx[PADDING_TOKEN]
        self.head_unk_id = head_dict.str2idx[UNKNOWN_TOKEN]
        self.char_padding_id = char_dict.str2idx[PADDING_TOKEN]
        self.char_unk_id = char_dict.str2idx[UNKNOWN_TOKEN]
        print "Word padding id {}, unk id {}".format(self.word_padding_id, self.word_unk_id)
        print "Head padding id {}, unk id {}".format(self.head_padding_id, self.head_unk_id)
        print "Char padding id {}, unk id {}".format(self.char_padding_id, self.char_unk_id)
        if embeddings is not None:
            self.word_embeddings = embeddings[0]
            self.head_embeddings = embeddings[1]
        if embeddings_shapes is not None:
            self.word_embedding_shapes = embeddings_shapes[0]
            self.head_embedding_shapes = embeddings_shapes[1]

    def get_corpus_predicates_Id(self, corpus):
        last_sentence_id = -1
        sentence_predicate_id = []
        for sentence in corpus:
            current_sentence_id = sentence[0]
            if last_sentence_id != current_sentence_id:
                sentence_predicate_id.append(sentence[0][2])
                last_sentence_id = current_sentence_id
        return sentence_predicate_id

    def tensorize_batch_samples(self, samples):
        """
        tensorize the batch samples
        :param samples: List of samples
        :return: tensorized batch samples
        """
        batch_sample_size = len(samples)
        max_sample_length = max([sam[1] for sam in samples])
        max_sample_word_length = max([sam[4].shape[1] for sam in samples])
        max_sample_arg_number = max([len(sam[5][0]) for sam in samples])
        max_sample_gold_predicates_number = max([sam[6][1] for sam in samples])
        # input
        padded_sample_ids = np.zeros(batch_sample_size, dtype=np.int64)
        padded_sample_lengths = np.zeros(batch_sample_size, dtype=np.int64)
        padded_word_tokens = np.zeros([batch_sample_size, max_sample_length], dtype=np.int64)
        padded_head_tokens = np.zeros([batch_sample_size, max_sample_length], dtype=np.int64)
        padded_char_tokens = np.zeros([batch_sample_size, max_sample_length, max_sample_word_length], dtype=np.int64)
        # gold predicates
        padded_gold_predicates = np.zeros([batch_sample_size, max_sample_gold_predicates_number], dtype=np.int64)
        padded_num_gold_predicates = np.zeros(batch_sample_size, dtype=np.int64)
        #
        if max_sample_arg_number == 0:
            max_sample_arg_number = 1
        padded_predicates = np.zeros([batch_sample_size, max_sample_arg_number], dtype=np.int64)
        padded_arg_starts = np.zeros([batch_sample_size, max_sample_arg_number], dtype=np.int64)  # one for padding the arg start
        padded_arg_ends = np.zeros([batch_sample_size, max_sample_arg_number], dtype=np.int64)  # zero for padding the arg end
        padded_arg_labels = np.zeros([batch_sample_size, max_sample_arg_number], dtype=np.int64)
        padded_srl_lens = np.zeros(batch_sample_size, dtype=np.int64)
        for i, sample in enumerate(samples):
            sample_length = sample[1]
            padded_sample_ids[i] = sample[0]
            # input
            padded_sample_lengths[i] = sample_length
            padded_word_tokens[i, :sample_length] = sample[2]
            padded_head_tokens[i, :sample_length] = sample[3]
            # print sample_length, len(sample[2]), len(sample[3]), padded_char_tokens.shape, sample[4].shape
            padded_char_tokens[i, :sample[4].shape[0], : sample[4].shape[1]] = sample[4]
            # gold predicates
            padded_gold_predicates[i, : sample[6][1]] = sample[6][0]
            padded_num_gold_predicates[i] = sample[6][1]
            # output
            sample_arg_number = len(sample[5][0])
            padded_predicates[i, :sample_arg_number] = sample[5][0]
            padded_arg_starts[i, :sample_arg_number] = sample[5][1]
            padded_arg_ends[i, :sample_arg_number] = sample[5][2]
            padded_arg_labels[i, :sample_arg_number] = sample[5][3]
            padded_srl_lens[i] = sample_arg_number
        return torch.from_numpy(padded_sample_ids), torch.from_numpy(padded_sample_lengths), \
               torch.from_numpy(padded_word_tokens), torch.from_numpy(padded_head_tokens), \
               torch.from_numpy(padded_char_tokens), \
               torch.from_numpy(padded_predicates), \
               torch.from_numpy(padded_arg_starts), torch.from_numpy(padded_arg_ends), \
               torch.from_numpy(padded_arg_labels), torch.from_numpy(padded_srl_lens), \
               torch.from_numpy(padded_gold_predicates), torch.from_numpy(padded_num_gold_predicates)

    def tensorize_dep_samples(self, samples):
        batch_num_samples = len(samples)
        max_sample_length = max([len(sam[0]) for sam in samples])
        max_sample_word_length = max([sam[1].shape[1] for sam in samples])
        # input
        padded_word_tokens = np.zeros([batch_num_samples, max_sample_length], dtype=np.int64)
        padded_char_tokens = np.zeros([batch_num_samples, max_sample_length, max_sample_word_length], dtype=np.int64)
        masks = np.zeros([batch_num_samples, max_sample_length], dtype=np.int64)
        lengths, heads, labels = [], [], []
        for i, sample in enumerate(samples):
            length = len(sample[0])
            lengths.append(length)
            # input
            padded_word_tokens[i, :length] = sample[0]
            padded_char_tokens[i, :sample[1].shape[0], :sample[1].shape[1]] = sample[1]
            masks[i, :length] = np.ones((length), dtype=np.int64)
            # output
            heads.append(sample[2])
            labels.append(sample[3])
        return torch.from_numpy(padded_word_tokens), torch.from_numpy(padded_char_tokens), torch.from_numpy(masks), \
               lengths, heads, labels

    def get_dep_training_data(self, include_last_batch=False):
        """
        :param include_last_batch:
        :return:
        """
        random.shuffle(self.dep_data)
        results, batch_tensors = [], []
        batch_num_tokens = 0
        for i, example in enumerate(self.dep_data):
            num_words = len(example[0])
            if len(batch_tensors) >= self.batch_size or batch_num_tokens + num_words >= self.max_tokens_per_batch:
                results.append(batch_tensors)
                batch_tensors = []
                batch_num_tokens = 0
            batch_tensors.append(example)
            batch_num_tokens += num_words

        results = [self.tensorize_dep_samples(batch_sample) for i, batch_sample in enumerate(results)]
        print("Extracted {} samples and {} batches.".format(len(self.dep_data), len(results)))
        return results

    def get_training_data(self, include_last_batch=False):
        """ Get shuffled training samples. Called at the beginning of each epoch.
        """
        # TODO: Speed up: Use variable size batches (different max length).
        random.shuffle(self.train_samples)  # TODO when the model is stable, uncomment it
        assert include_last_batch is True

        num_samples = len(self.train_samples)
        results, batch_tensors = [], []
        batch_num_tokens = 0
        for i, example in enumerate(self.train_samples):
            num_words = example[1]
            if len(batch_tensors) >= self.batch_size or batch_num_tokens + num_words >= self.max_tokens_per_batch:
                results.append(batch_tensors)
                batch_tensors = []
                batch_num_tokens = 0
            batch_tensors.append(example)
            batch_num_tokens += num_words

        results = [self.tensorize_batch_samples(batch_sample) for i, batch_sample in enumerate(results)]
        print("Extracted {} samples and {} batches.".format(num_samples, len(results)))
        return results

    @staticmethod
    def mix_training_data(srl_data, dep_data):
        max_batch_size = max(len(srl_data), len(dep_data))
        min_batch_size = min(len(srl_data), len(dep_data))
        multiple = int(max_batch_size / min_batch_size)
        # ok, i know the dep_data is small_size
        which_bigger = 0 if len(srl_data) > len(dep_data) else 1
        if which_bigger == 0:
            # it's impossible
            exit()
        else:
            results = zip(srl_data * multiple, dep_data)  # the order is sensitive
            print("After mix operation, the resulting training batch size is {}".format(len(results)))
            return results

    def get_development_data(self, batch_size=None):
        if batch_size is None:
            return self.dev_samples

        num_samples = len(self.dev_samples)
        batched_tensors = [self.dev_samples[i: min(i + batch_size, num_samples)]
                           for i in xrange(0, num_samples, batch_size)]
        results = [self.tensorize_batch_samples(t) for t in batched_tensors]
        return results

    def get_test_data(self, test_sentences, batch_size=None):
        num_samples = len(test_sentences)
        batched_tensors = [test_sentences[i: min(i + batch_size, num_samples)]
                           for i in xrange(0, num_samples, batch_size)]
        if batch_size is None:
            return batched_tensors
        results = [self.tensorize_batch_samples(t) for t in batched_tensors]
        return results
