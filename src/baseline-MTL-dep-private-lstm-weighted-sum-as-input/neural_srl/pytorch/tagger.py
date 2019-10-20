import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from layer import NonLinear, Biaffine
from HighWayLSTM import *


def initializer_1d(input_tensor, initializer):
    assert len(input_tensor.size()) == 1
    input_tensor = input_tensor.view(-1, 1)
    input_tensor = initializer(input_tensor)
    return input_tensor.view(-1)


def pad_sequence(xs, length=None, padding=-1, dtype=np.float64):
    lengths = [len(x) for x in xs]
    if length is None:
        length = max(lengths)
    y = np.array([np.pad(x.astype(dtype), (0, length - l),
                         mode="constant", constant_values=padding)
                  for x, l in zip(xs, lengths)])
    return torch.from_numpy(y)


def _model_var(parameters, x):
    p = next(iter(filter(lambda p: p.requires_grad, parameters)))
    if p.is_cuda:
        x = x.cuda(p.get_device())
    return torch.autograd.Variable(x)


class BiLSTMTaggerModel(nn.Module):
    """ Constructs the network and builds the following Theano functions:
        - pred_function: Takes input and mask, returns prediction.
        - loss_function: Takes input, mask and labels, returns the final cross entropy loss (scalar).
    """
    def __init__(self, data, config, gpu_id=""):
        super(BiLSTMTaggerModel, self).__init__()
        self.config = config
        self.dropout = float(config.dropout)  # 0.2
        self.lexical_dropout = float(self.config.lexical_dropout)
        self.lstm_type = config.lstm_cell
        self.lstm_hidden_size = int(config.lstm_hidden_size)  # SRL: 300
        self.num_lstm_layers = int(config.num_lstm_layers)  # SRL:
        self.max_grad_norm = float(config.max_grad_norm)
        self.use_gold_predicates = config.use_gold_predicates
        print("USE_GOLD_PREDICATES ? ", self.use_gold_predicates)

        self.word_embedding_shapes = data.word_embedding_shapes
        self.head_embedding_shapes = data.head_embedding_shapes
        self.vocab_size = data.word_dict.size()
        self.head_size = data.head_dict.size()
        self.char_size = data.char_dict.size()
        self.label_space_size = data.label_dict.size()
        self.dep_label_space_size = data.dep_label_dict.size()
        self.cuda_id = gpu_id

        # Initialize layers and parameters
        word_embedding_shape = self.word_embedding_shapes
        assert word_embedding_shape[0] == self.vocab_size
        self.word_embedding_dim = word_embedding_shape[1]  # get the embedding dim
        self.word_embedding = nn.Embedding(word_embedding_shape[0], self.word_embedding_dim, padding_idx=0)

        head_embedding_shape = self.head_embedding_shapes
        self.head_embedding_dim = head_embedding_shape[1]
        self.head_embedding = nn.Embedding(head_embedding_shape[0], self.head_embedding_dim, padding_idx=0)
        # character embedding
        self.char_embedding = nn.Embedding(self.char_size, self.config.char_emb_size, padding_idx=0)

        self.word_embedding.weight.data.copy_(torch.from_numpy(data.word_embeddings))
        self.word_embedding.weight.requires_grad = False
        self.head_embedding.weight.data.copy_(torch.from_numpy(data.head_embeddings))
        self.head_embedding.weight.requires_grad = False
        # char cnn layer
        # Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1...
        self.char_cnns = nn.ModuleList([nn.Conv1d(self.config.char_emb_size, self.config.output_channel, int(kernel_size),
                                                  stride=1, padding=0) for kernel_size in self.config.kernel_sizes])
        # softmax weights
        self.gamma = nn.Parameter(torch.FloatTensor([1.0]))
        self.softmax_dep_weights = nn.ParameterList([nn.Parameter(torch.FloatTensor([0.0]))
                                                     for _ in range(self.num_lstm_layers)])
        # Initialize HighwayBiLSTM
        self.lstm_input_size = self.word_embedding_dim + 3 * self.config.output_channel  # word emb dim + char cnn dim
        self.bilstm = Highway_Concat_BiLSTM(
            input_size=self.lstm_input_size + 2 * self.lstm_hidden_size,
            hidden_size=self.lstm_hidden_size,  # // 2 for MyLSTM
            num_layers=self.num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout_in=config.input_dropout_prob,
            dropout_out=config.recurrent_dropout_prob
        )
        self.dep_bilstm = Highway_Concat_BiLSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.lstm_hidden_size,  # // 2 for MyLSTM
            num_layers=self.num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout_in=config.input_dropout_prob,
            dropout_out=config.recurrent_dropout_prob
        )
        # span width feature embedding
        self.span_width_embedding = nn.Embedding(self.config.max_arg_width, self.config.span_width_feature_size)
        self.context_projective_layer = nn.Linear(2 * self.lstm_hidden_size, self.config.num_attention_heads)
        # span scores
        self.span_emb_size = 2 * 2 * self.lstm_hidden_size + 2 * self.lstm_hidden_size + self.config.span_width_feature_size
        self.arg_unary_score_layers = nn.ModuleList([nn.Linear(self.span_emb_size, self.config.ffnn_size) if i == 0
                                                     else nn.Linear(self.config.ffnn_size, self.config.ffnn_size) for i
                                                     in range(self.config.ffnn_depth)])  #[,150]
        self.arg_dropout_layers = nn.ModuleList([nn.Dropout(self.dropout) for _ in range(self.config.ffnn_depth)])
        self.arg_unary_score_projection = nn.Linear(self.config.ffnn_size, 1)
        # predicate scores
        self.pred_unary_score_layers = nn.ModuleList([nn.Linear(2 * self.lstm_hidden_size, self.config.ffnn_size) if i == 0
                        else nn.Linear(self.config.ffnn_size, self.config.ffnn_size) for i
                        in range(self.config.ffnn_depth)])  # [,150]
        self.pred_dropout_layers = nn.ModuleList([nn.Dropout(self.dropout) for _ in range(self.config.ffnn_depth)])
        self.pred_unary_score_projection = nn.Linear(self.config.ffnn_size, 1)
        # srl scores
        self.srl_unary_score_input_size = self.span_emb_size + 2 * self.lstm_hidden_size
        self.srl_unary_score_layers = nn.ModuleList([nn.Linear(self.srl_unary_score_input_size, self.config.ffnn_size)
                                       if i == 0 else nn.Linear(self.config.ffnn_size, self.config.ffnn_size)
                                                     for i in range(self.config.ffnn_depth)])
        self.srl_dropout_layers = nn.ModuleList([nn.Dropout(self.dropout) for _ in range(self.config.ffnn_depth)])
        self.srl_unary_score_projection = nn.Linear(self.config.ffnn_size, self.label_space_size - 1)

        # dependency parsing module
        self.mlp_arc_dep = NonLinear(
            input_size=2*config.lstm_hidden_size,
            hidden_size=config.mlp_arc_size + config.mlp_rel_size,
            activation=nn.LeakyReLU(0.1))
        self.mlp_arc_head = NonLinear(
            input_size=2*config.lstm_hidden_size,
            hidden_size=config.mlp_arc_size + config.mlp_rel_size,
            activation=nn.LeakyReLU(0.1))

        self.total_num = int((config.mlp_arc_size + config.mlp_rel_size) / 100)
        self.arc_num = int(config.mlp_arc_size / 100)
        self.rel_num = int(config.mlp_rel_size / 100)

        self.arc_biaffine = Biaffine(config.mlp_arc_size, config.mlp_arc_size, 1, bias=(True, False))
        self.rel_biaffine = Biaffine(config.mlp_rel_size, config.mlp_rel_size, self.dep_label_space_size,
                                     bias=(True, True))

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.char_embedding.weight)

        for layer in self.char_cnns:
            init.xavier_uniform_(layer.weight)
            initializer_1d(layer.bias, init.xavier_uniform_)

        init.xavier_uniform_(self.span_width_embedding.weight)
        init.xavier_uniform_(self.context_projective_layer.weight)
        initializer_1d(self.context_projective_layer.bias, init.xavier_uniform_)

        for layer in self.arg_unary_score_layers:
            init.xavier_uniform_(layer.weight)
            initializer_1d(layer.bias, init.xavier_uniform_)
        init.xavier_uniform_(self.arg_unary_score_projection.weight)
        initializer_1d(self.arg_unary_score_projection.bias, init.xavier_uniform_)

        for layer in self.pred_unary_score_layers:
            init.xavier_uniform_(layer.weight)
            initializer_1d(layer.bias, init.xavier_uniform_)
        init.xavier_uniform_(self.pred_unary_score_projection.weight)
        initializer_1d(self.pred_unary_score_projection.bias, init.xavier_uniform_)

        for layer in self.srl_unary_score_layers:
            init.xavier_uniform_(layer.weight)
            initializer_1d(layer.bias, init.xavier_uniform_)
        init.xavier_uniform_(self.srl_unary_score_projection.weight)
        initializer_1d(self.srl_unary_score_projection.bias, init.xavier_uniform_)
        return None

    def init_masks(self, batch_size, lengths):
        max_sent_length = max(lengths)
        num_sentences = batch_size
        indices = torch.arange(0, max_sent_length).unsqueeze(0).expand(num_sentences, -1)
        masks = indices < lengths.unsqueeze(1)
        masks = masks.type(torch.FloatTensor)
        if self.cuda_id:
            masks = masks.cuda()
        return masks

    def sequence_mask(self, sent_lengths, max_sent_length=None):
        batch_size, max_length = sent_lengths.size()[0], torch.max(sent_lengths)
        if max_sent_length is not None:
            max_length = max_sent_length
        indices = torch.arange(0, max_length).unsqueeze(0).expand(batch_size, -1)
        mask = indices < sent_lengths.unsqueeze(1).cpu()
        mask = mask.type(torch.LongTensor)
        if torch.cuda.is_available():
            mask = mask.cuda()
        return mask

    def get_char_cnn_embeddings(self, chars):
        num_sentences, max_sentence_length = chars.size()[0], chars.size()[1]
        chars_embeddings = self.char_embedding(chars)

        chars_embeddings = chars_embeddings.view(num_sentences * max_sentence_length,
                                                 chars_embeddings.size()[2], chars_embeddings.size()[3])
        # [Batch_size, Input_size, Seq_len]
        chars_embeddings = chars_embeddings.transpose(1, 2)
        chars_output = []
        for i, cnn in enumerate(self.char_cnns):
            chars_cnn_embedding = F.relu(cnn.forward(chars_embeddings))
            pooled_chars_cnn_emb, _ = chars_cnn_embedding.max(2)
            chars_output.append(pooled_chars_cnn_emb)
        chars_output_emb = torch.cat(chars_output, 1)
        return chars_output_emb.view(num_sentences, max_sentence_length, chars_output_emb.size()[1])

    @staticmethod
    def get_candidate_spans(sent_lengths, max_sent_length, max_arg_width):
        num_sentences = len(sent_lengths)
        candidate_starts = torch.arange(0, max_sent_length).expand(num_sentences, max_arg_width, -1)
        candidate_width = torch.arange(0, max_arg_width).view(1, -1, 1)
        candidate_ends = candidate_starts + candidate_width

        candidate_starts = candidate_starts.contiguous().view(num_sentences, max_sent_length * max_arg_width)
        candidate_ends = candidate_ends.contiguous().view(num_sentences, max_sent_length * max_arg_width)
        actual_sent_lengths = sent_lengths.view(-1, 1).expand(-1, max_sent_length * max_arg_width)
        candidate_mask = candidate_ends < actual_sent_lengths.type(torch.LongTensor)
        float_candidate_mask = candidate_mask.type(torch.LongTensor)

        candidate_starts = candidate_starts * float_candidate_mask
        candidate_ends = candidate_ends * float_candidate_mask
        return candidate_starts, candidate_ends, candidate_mask

    @staticmethod
    def exclusive_cumsum(input, exclusive=True):
        """
        :param input: input is the sentence lengths tensor.
        :param exclusive: exclude the last sentence length
        :return: the sum of y_i = x_1 + x_2 + ... + x_{i - 1} (i >= 1, and x_0 = 0)
        """
        assert exclusive is True
        if exclusive is True:
            exclusive_sent_lengths = torch.zeros(1).type(torch.LongTensor)
            result = torch.cumsum(torch.cat([exclusive_sent_lengths, input], 0)[:-1], 0).view(-1, 1)
        else:
            result = torch.cumsum(input, 0).view(-1, 1)
        return result

    def flatten_emb(self, emb):
        num_sentences, max_sentence_length = emb.size()[0], emb.size()[1]
        assert len(emb.size()) == 3
        flatted_emb = emb.contiguous().view(num_sentences * max_sentence_length, -1)
        return flatted_emb

    def flatten_emb_in_sentence(self, emb, batch_sentences_mask):
        num_sentences, max_sentence_length = emb.size()[0], emb.size()[1]
        flatted_emb = self.flatten_emb(emb)
        return flatted_emb[batch_sentences_mask.view(num_sentences * max_sentence_length)]

    def get_span_emb(self, flatted_head_emb, flatted_context_emb, flatted_candidate_starts, flatted_candidate_ends,
                     config, dropout=0.0):
        batch_word_num = flatted_context_emb.size()[0]
        span_num = flatted_candidate_starts.size()[0]  # candidate span num.
        # gather slices from embeddings according to indices
        span_start_emb = flatted_context_emb[flatted_candidate_starts]
        span_end_emb = flatted_context_emb[flatted_candidate_ends]
        span_emb_feature_list = [span_start_emb, span_end_emb]  # store the span vector representations for span rep.

        span_width = 1.0 + flatted_candidate_ends - flatted_candidate_starts  # [num_spans], generate the span width
        max_arg_width = config.max_arg_width
        num_heads = config.num_attention_heads

        # get the span width feature emb
        span_width_index = span_width - 1
        span_width_emb = self.span_width_embedding(span_width_index.cuda())
        span_width_emb = F.dropout(span_width_emb, dropout, self.training)
        span_emb_feature_list.append(span_width_emb)

        """head features"""
        cpu_flatted_candidte_starts = flatted_candidate_starts.cpu()
        span_indices = torch.arange(0, max_arg_width).type(torch.LongTensor).view(1, -1) + \
                       cpu_flatted_candidte_starts.view(-1, 1)  # For all the i, where i in [begin, ..i, end] for span
        # reset the position index to the batch_word_num index with index - 1
        span_indices = torch.clamp(span_indices, max=batch_word_num - 1)
        num_spans, spans_width = span_indices.size()[0], span_indices.size()[1]
        flatted_span_indices = span_indices.view(-1)  # so Huge!!!, column is the span?
        # if torch.cuda.is_available():
        flatted_span_indices = flatted_span_indices.cuda()
        span_text_emb = flatted_context_emb.index_select(0, flatted_span_indices).view(num_spans, spans_width, -1)
        span_indices_mask = self.sequence_mask(span_width, max_sent_length=max_arg_width)
        span_indices_mask = span_indices_mask.type(torch.Tensor).cuda()
        # project context output to num head
        head_scores = self.context_projective_layer.forward(flatted_context_emb)
        # get span attention
        # span_attention = head_scores.index_select(0, flatted_span_indices).view(num_spans, spans_width)
        # span_attention = torch.add(span_attention, expanded_span_indices_log_mask).unsqueeze(2)  # control the span len
        # span_attention = F.softmax(span_attention, dim=1)
        span_text_emb = span_text_emb * span_indices_mask.unsqueeze(2).\
            expand(-1, -1, span_text_emb.size()[-1])
        span_head_emb = torch.mean(span_text_emb, 1)
        span_emb_feature_list.append(span_head_emb)

        span_emb = torch.cat(span_emb_feature_list, 1)
        return span_emb, head_scores, span_text_emb, span_indices, span_indices_mask

    def get_arg_unary_scores(self, span_emb, config, dropout, num_labels=1, name="span_scores"):
        """
        Compute span score with FFNN(span embedding)
        :param span_emb: tensor of [num_sentences, num_spans, emb_size]
        :param config:
        :param dropout:
        :param num_labels:
        :param name:
        :return:
        """
        input = span_emb
        for i, ffnn in enumerate(self.arg_unary_score_layers):
            input = F.relu(ffnn.forward(input))
            input = self.arg_dropout_layers[i].forward(input)
        output = self.arg_unary_score_projection.forward(input)
        return output

    def get_pred_unary_scores(self, span_emb, config, dropout, num_labels=1, name="pred_scores"):
        input = span_emb
        for i, ffnn in enumerate(self.pred_unary_score_layers):
            input = F.relu(ffnn.forward(input))
            input = self.pred_dropout_layers[i].forward(input)
        output = self.pred_unary_score_projection.forward(input)
        return output

    def extract_spans(self, candidate_scores, candidate_starts, candidate_ends, topk, max_sentence_length,
                           sort_spans, enforce_non_crossing):
        """
        extract the topk span indices
        :param candidate_scores:
        :param candidate_starts:
        :param candidate_ends:
        :param topk: [num_sentences]
        :param max_sentence_length:
        :param sort_spans:
        :param enforce_non_crossing:
        :return: indices [num_sentences, max_num_predictions]
        """
        num_sentences = candidate_scores.size()[0]
        num_input_spans = candidate_scores.size()[1]
        max_num_output_spans = int(torch.max(topk))
        indices = [score.topk(k)[1] for score, k in zip(candidate_scores, topk)]
        output_span_indices_tensor = [F.pad(item, [0, max_num_output_spans - item.size()[0]], value=item[-1])
                                      for item in indices]
        output_span_indices_tensor = torch.stack(output_span_indices_tensor).cpu()
        return output_span_indices_tensor

    def batch_index_select(self, emb, indices):
        num_sentences = emb.size()[0]
        max_sent_length = emb.size()[1]
        flatten_emb = self.flatten_emb(emb)
        offset = (torch.arange(0, num_sentences) * max_sent_length).unsqueeze(1).cuda()
        return torch.index_select(flatten_emb, 0, (indices + offset).view(-1))\
            .view(indices.size()[0], indices.size()[1], -1)

    def get_batch_topk(self, candidate_starts, candidate_ends, candidate_scores, topk_ratio, text_len,
                       max_sentence_length, sort_spans=False, enforce_non_crossing=True):
        num_sentences = candidate_starts.size()[0]
        max_sentence_length = candidate_starts.size()[1]

        topk = torch.floor(text_len.type(torch.FloatTensor) * topk_ratio)
        topk = torch.max(topk.type(torch.LongTensor), torch.ones(num_sentences).type(torch.LongTensor))

        # this part should be implemented with C++
        predicted_indices = self.extract_spans(candidate_scores, candidate_starts, candidate_ends, topk,
                                               max_sentence_length, sort_spans, enforce_non_crossing)
        predicted_starts = torch.gather(candidate_starts, 1, predicted_indices)
        predicted_ends = torch.gather(candidate_ends, 1, predicted_indices)
        predicted_scores = torch.gather(candidate_scores, 1, predicted_indices.cuda())
        return predicted_starts, predicted_ends, predicted_scores, topk, predicted_indices

    def get_dense_span_labels(self, span_starts, span_ends, span_labels, num_spans, max_sentence_length, span_parents=None):
        num_sentences = span_starts.size()[0]
        max_spans_num = span_starts.size()[1]

        span_starts = span_starts + 1 - self.sequence_mask(num_spans)
        sentence_indices = torch.arange(0, num_sentences).unsqueeze(1).expand(-1, max_spans_num).type(torch.LongTensor).cuda()

        sparse_indices = torch.cat([sentence_indices.unsqueeze(2), span_starts.unsqueeze(2), span_ends.unsqueeze(2)],
                                   dim=2)
        if span_parents is not None:  # semantic span labels
            sparse_indices = torch.cat([sparse_indices, span_parents.unsqueeze(2)], 2)

        rank = 3 if span_parents is None else 4
        dense_labels = torch.sparse.FloatTensor(sparse_indices.cpu().view(num_sentences * max_spans_num, rank).t(),
                                                span_labels.view(-1).type(torch.FloatTensor),
                                                torch.Size([num_sentences] + [max_sentence_length] * (rank - 1)))\
            .to_dense()  # ok @kiro
        return dense_labels

    def gather_4d(self, params, indices):
        assert len(params.size()) == 4 and len(indices.size()) == 4
        params = params.type(torch.LongTensor)
        indices_a, indices_b, indices_c, indices_d = indices.chunk(4, dim=3)
        result = params[indices_a, indices_b, indices_c, indices_d]
        return result.unsqueeze(3)

    def get_srl_labels(self, arg_starts, arg_ends, predicates, labels, max_sentence_length):
        num_sentences = arg_starts.size()[0]
        max_arg_num = arg_starts.size()[1]
        max_pred_num = predicates.size()[1]

        sentence_indices_2d = torch.arange(0, num_sentences).unsqueeze(1).unsqueeze(2).expand(-1, max_arg_num, max_pred_num)
        expanded_arg_starts = arg_starts.unsqueeze(2).expand(-1, -1, max_pred_num)
        expanded_arg_ends = arg_ends.unsqueeze(2).expand(-1, -1, max_pred_num)
        expanded_predicates = predicates.unsqueeze(1).expand(-1, max_arg_num, -1)

        pred_indices = torch.cat([sentence_indices_2d.unsqueeze(3), expanded_arg_starts.unsqueeze(3),
                                  expanded_arg_ends.unsqueeze(3), expanded_predicates.unsqueeze(3)], 3)

        dense_srl_labels = self.get_dense_span_labels(labels[1], labels[2], labels[3], labels[4],
                                                      max_sentence_length, span_parents=labels[0])  # ans
        srl_labels = self.gather_4d(dense_srl_labels, pred_indices.type(torch.LongTensor))  # TODO !!!!!!!!!!!!!
        return srl_labels

    def get_srl_unary_scores(self, span_emb, config, dropout, num_labels=1, name="span_scores"):
        input = span_emb
        for i, ffnn in enumerate(self.srl_unary_score_layers):
            input = F.relu(ffnn.forward(input))
            # if self.training:
            input = self.srl_dropout_layers[i].forward(input)
        output = self.srl_unary_score_projection.forward(input)
        return output

    def get_srl_scores(self, arg_emb, pred_emb, arg_scores, pred_scores, num_labels, config, dropout):
        num_sentences = arg_emb.size()[0]
        num_args = arg_emb.size()[1]  # [batch_size, max_arg_num, arg_emb_size]
        num_preds = pred_emb.size()[1]  # [batch_size, max_pred_num, pred_emb_size]

        unsqueezed_arg_emb = arg_emb.unsqueeze(2)
        unsqueezed_pred_emb = pred_emb.unsqueeze(1)
        expanded_arg_emb = unsqueezed_arg_emb.expand(-1, -1, num_preds, -1)
        expanded_pred_emb = unsqueezed_pred_emb.expand(-1, num_args, -1, -1)
        pair_emb_list = [expanded_arg_emb, expanded_pred_emb]
        pair_emb = torch.cat(pair_emb_list, 3)  # concatenate the argument emb and pre emb
        pair_emb_size = pair_emb.size()[3]
        flat_pair_emb = pair_emb.view(num_sentences * num_args * num_preds, pair_emb_size)
        # get unary scores
        flat_srl_scores = self.get_srl_unary_scores(flat_pair_emb, config, dropout, num_labels - 1,
                                                    "predicate_argument_scores")
        srl_scores = flat_srl_scores.view(num_sentences, num_args, num_preds, -1)
        unsqueezed_arg_scores, unsqueezed_pred_scores = \
            arg_scores.unsqueeze(2).unsqueeze(3), pred_scores.unsqueeze(1).unsqueeze(3)  # TODO ?
        srl_scores = srl_scores + unsqueezed_arg_scores + unsqueezed_pred_scores
        dummy_scores = torch.zeros([num_sentences, num_args, num_preds, 1]).cuda()
        srl_scores = torch.cat([dummy_scores, srl_scores], 3)
        return srl_scores

    def get_srl_softmax_loss(self, srl_scores, srl_labels, num_predicted_args, num_predicted_preds):
        max_num_arg = srl_scores.size()[1]
        max_num_pred = srl_scores.size()[2]
        num_labels = srl_scores.size()[3]

        # num_predicted_args, 1D tensor; max_num_arg: a int variable means the gold ans's max arg number
        args_mask = self.sequence_mask(num_predicted_args, max_num_arg)
        pred_mask = self.sequence_mask(num_predicted_preds, max_num_pred)
        srl_loss_mask = Variable((args_mask.unsqueeze(2) == 1) & (pred_mask.unsqueeze(1) == 1))

        srl_scores = srl_scores.view(-1, num_labels)
        srl_labels = Variable(srl_labels.view(-1, 1)).cuda()
        output = F.log_softmax(srl_scores, 1)

        negative_log_likelihood_flat = -torch.gather(output, dim=1, index=srl_labels).view(-1)
        srl_loss_mask = (srl_loss_mask.view(-1) == 1).nonzero().view(-1)
        negative_log_likelihood = torch.gather(negative_log_likelihood_flat, dim=0, index=srl_loss_mask)
        loss = negative_log_likelihood.sum()  # origin is sum @kiro
        return loss

    def compute_dep_loss(self, true_arcs, true_rels, lengths):
        b, l1, l2 = self.arc_logits.size()
        index_true_arcs = _model_var(
            self.parameters(),
            pad_sequence(true_arcs, length=l1, padding=0, dtype=np.int64))
        true_arcs = _model_var(
            self.parameters(),
            pad_sequence(true_arcs, length=l1, padding=-1, dtype=np.int64))

        masks = []
        for length in lengths:
            mask = torch.FloatTensor([0] * length + [-1000] * (l2 - length))
            mask = _model_var(self.parameters(), mask)
            mask = torch.unsqueeze(mask, dim=1).expand(-1, l1)
            masks.append(mask.transpose(0, 1))
        length_mask = torch.stack(masks, 0)
        arc_logits = self.arc_logits + length_mask

        arc_loss = F.cross_entropy(
            arc_logits.view(b * l1, l2), true_arcs.view(b * l1),
            ignore_index=-1, reduction="sum")

        size = self.rel_logits.size()
        output_logits = _model_var(self.parameters(), torch.zeros(size[0], size[1], size[3]))

        for batch_index, (logits, arcs) in enumerate(zip(self.rel_logits, index_true_arcs)):
            rel_probs = []
            for i in range(l1):
                rel_probs.append(logits[i][int(arcs[i])])
            rel_probs = torch.stack(rel_probs, dim=0)
            output_logits[batch_index] = torch.squeeze(rel_probs, dim=1)

        b, l1, d = output_logits.size()
        true_rels = _model_var(self.parameters(), pad_sequence(true_rels, padding=-1, dtype=np.int64))

        rel_loss = F.cross_entropy(
            output_logits.view(b * l1, d), true_rels.view(b * l1), ignore_index=-1, reduction="sum")

        loss = arc_loss + rel_loss

        return loss

    def forward(self, sent_lengths, words, heads, chars, labels, gold_predicates=None, dep=None):
        num_sentences, max_sent_length = words.size()[0], words.size()[1]
        if dep is not None:
            word_embeddings, char_embeddings = \
                self.word_embedding(words), self.get_char_cnn_embeddings(chars)

            context_embeddings = torch.cat([word_embeddings, char_embeddings], dim=2)
            context_embeddings = F.dropout(context_embeddings, self.lexical_dropout, self.training)

            masks = self.init_masks(num_sentences, torch.LongTensor(sent_lengths))
            lstm_out, _, _ = self.dep_bilstm(context_embeddings, masks)

            if self.training:
                lstm_out = drop_sequence_sharedmask(lstm_out, self.config.dropout_mlp)

            x_all_dep = self.mlp_arc_dep(lstm_out)
            x_all_head = self.mlp_arc_head(lstm_out)

            if self.training:
                x_all_dep = drop_sequence_sharedmask(x_all_dep, self.config.dropout_mlp)
                x_all_head = drop_sequence_sharedmask(x_all_head, self.config.dropout_mlp)

            x_all_dep_splits = torch.split(x_all_dep, 100, dim=2)
            x_all_head_splits = torch.split(x_all_head, 100, dim=2)

            x_arc_dep = torch.cat(x_all_dep_splits[:self.arc_num], dim=2)
            x_arc_head = torch.cat(x_all_head_splits[:self.arc_num], dim=2)

            arc_logit = self.arc_biaffine(x_arc_dep, x_arc_head)
            arc_logit = torch.squeeze(arc_logit, dim=3)

            x_rel_dep = torch.cat(x_all_dep_splits[self.arc_num:], dim=2)
            x_rel_head = torch.cat(x_all_head_splits[self.arc_num:], dim=2)

            rel_logit_cond = self.rel_biaffine(x_rel_dep, x_rel_head)

            self.arc_logits, self.rel_logits = arc_logit, rel_logit_cond

            heads, rels = dep[0], dep[1]
            loss = self.compute_dep_loss(heads, rels, sent_lengths)  # compute the dep loss
            return loss

        word_embeddings, head_embeddings, char_embeddings = \
            self.word_embedding(words), self.head_embedding(heads), self.get_char_cnn_embeddings(chars)

        context_embeddings = torch.cat([word_embeddings, char_embeddings], dim=2)
        head_embeddings = torch.cat([head_embeddings, char_embeddings], dim=2)
        context_embeddings, head_embeddings = F.dropout(context_embeddings, self.lexical_dropout, self.training), \
                                              F.dropout(head_embeddings, self.lexical_dropout, self.training)

        masks = self.init_masks(num_sentences, sent_lengths)
        # dep lstm
        dep_lstm_out, _, dep_lstm_outputs = self.dep_bilstm(context_embeddings, masks)
        normed_weights = F.softmax(torch.cat([param for param in self.softmax_dep_weights]), dim=0)
        normed_weights = torch.split(normed_weights, 1)  # split_size_or_sections=1, split_size=1)  # 0.3.0
        dep_representations = self.gamma * (normed_weights[0] * dep_lstm_outputs[0] + normed_weights[1] * dep_lstm_outputs[1]
                                            + normed_weights[2] * dep_lstm_outputs[2])
        context_embeddings = torch.cat([context_embeddings, dep_representations], dim=-1)
        lstm_out, _, _ = self.bilstm(context_embeddings, masks)

        """generate candidate spans with argument pruning"""
        # candidate_starts [num_sentences, max_sent_length * max_arg_width]
        candidate_starts, candidate_ends, candidate_mask = BiLSTMTaggerModel.get_candidate_spans(
            sent_lengths, max_sent_length, self.config.max_arg_width)
        flatted_candidate_mask = candidate_mask.view(-1)
        batch_word_offset = BiLSTMTaggerModel.exclusive_cumsum(sent_lengths)  # get the word offset in a batch
        # choose the flatted_candidate_starts with the actual existing positions, i.e. exclude the illegal starts
        flatted_candidate_starts = candidate_starts + batch_word_offset
        flatted_candidate_starts = flatted_candidate_starts.view(-1)[flatted_candidate_mask].type(torch.LongTensor)
        flatted_candidate_ends = candidate_ends + batch_word_offset
        flatted_candidate_ends = flatted_candidate_ends.view(-1)[flatted_candidate_mask].type(torch.LongTensor)
        # flatten the lstm output according to the sentence mask, i.e. exclude the illegal (padding) lstm output
        byte_sentence_mask = masks.type(torch.ByteTensor).cuda()  # convert cpu type to cuda type
        flatted_context_output = self.flatten_emb_in_sentence(lstm_out, byte_sentence_mask)  # cuda type
        flatted_head_emb = self.flatten_emb_in_sentence(head_embeddings, byte_sentence_mask)
        batch_word_num = flatted_context_output.size()[0]  # total word num in batch sentences

        """generate the span embedding"""
        if torch.cuda.is_available():
            flatted_candidate_starts, flatted_candidate_ends = \
                flatted_candidate_starts.cuda(), flatted_candidate_ends.cuda()
        candidate_span_emb, head_scores, span_head_emb, head_indices, head_indices_log_mask = self.get_span_emb(
            flatted_head_emb, flatted_context_output, flatted_candidate_starts, flatted_candidate_ends,
            self.config, dropout=self.dropout)
        """Get the span ids"""
        candidate_span_number = candidate_span_emb.size()[0]
        max_candidate_spans_num_per_sentence = candidate_mask.size()[1]
        sparse_indices = (candidate_mask == 1).nonzero()
        sparse_values = torch.arange(0, candidate_span_number)
        candidate_span_ids = torch.sparse.FloatTensor(sparse_indices.t(), sparse_values,
                                        torch.Size([num_sentences, max_candidate_spans_num_per_sentence])).to_dense()
        candidate_span_ids = candidate_span_ids.type(torch.LongTensor).cuda()  # ok @kiro

        spans_log_mask = torch.log(candidate_mask.type(torch.Tensor)).cuda()
        predict_dict = {"candidate_starts": candidate_starts, "candidate_ends": candidate_ends, "head_scores": head_scores}

        """Get unary scores and topk of candidate argument spans."""
        flatted_candidate_arg_scores = self.get_arg_unary_scores(candidate_span_emb, self.config, self.dropout,
                                                                 1, "argument scores")
        candidate_arg_scores = flatted_candidate_arg_scores.index_select(0, candidate_span_ids.view(-1))\
            .view(candidate_span_ids.size()[0], candidate_span_ids.size()[1])
        candidate_arg_scores = candidate_arg_scores + spans_log_mask
        arg_starts, arg_ends, arg_scores, num_args, top_arg_indices = \
            self.get_batch_topk(candidate_starts, candidate_ends, candidate_arg_scores,
                                self.config.argument_ratio, sent_lengths, max_sent_length,
                                sort_spans=False, enforce_non_crossing=False)
        """Get the candidate predicate"""
        candidate_pred_ids = torch.arange(0, max_sent_length).unsqueeze(0).expand(num_sentences, -1)
        candidate_pred_emb = lstm_out
        candidate_pred_scores = self.get_pred_unary_scores(candidate_pred_emb, self.config, self.dropout,
                                                           1, "pred_scores")
        candidate_pred_scores = candidate_pred_scores + torch.log(masks.type(torch.Tensor).unsqueeze(2)).cuda()
        candidate_pred_scores = candidate_pred_scores.squeeze(2)

        if self.use_gold_predicates is True:
            predicates = gold_predicates[0]
            num_preds = gold_predicates[1]
            pred_scores = torch.zeros_like(predicates).type(torch.FloatTensor).cuda()
            top_pred_indices = predicates
        else:
            predicates, _, pred_scores, num_preds, top_pred_indices = self.get_batch_topk(
                candidate_pred_ids, candidate_pred_ids, candidate_pred_scores, self.config.predicate_ratio,
                sent_lengths, max_sent_length,
                sort_spans=False, enforce_non_crossing=False)

        """Get top arg embeddings"""
        arg_span_indices = torch.gather(candidate_span_ids, 1, top_arg_indices.cuda())  # [num_sentences, max_num_args]
        arg_emb = candidate_span_emb.index_select(0, arg_span_indices.view(-1)).view(
            arg_span_indices.size()[0], arg_span_indices.size()[1], -1
        )  # [num_sentences, max_num_args, emb]
        """Get top predicate embeddings"""
        pred_emb = self.batch_index_select(candidate_pred_emb, top_pred_indices.cuda())  # [num_sentences, max_num_preds, emb]

        max_arg_num = arg_scores.size()[1]
        max_pred_num = pred_scores.size()[1]

        """Get the answers according to the labels"""
        srl_labels = self.get_srl_labels(arg_starts, arg_ends, predicates, labels, max_sent_length)
        """Get the srl scores according to the arg emb and pre emb."""
        srl_scores = self.get_srl_scores(arg_emb, pred_emb, arg_scores, pred_scores, self.label_space_size, self.config,
                                         self.dropout)  # [num_sentences, max_num_args, max_num_preds, num_labels]
        """Compute the srl loss"""
        srl_loss = self.get_srl_softmax_loss(srl_scores, srl_labels, num_args, num_preds)

        predict_dict.update({
            "candidate_arg_scores": candidate_arg_scores,
            "candidate_pred_scores": candidate_pred_scores,
            "arg_starts": arg_starts,
            "arg_ends": arg_ends,
            "predicates": predicates,
            "arg_scores": arg_scores,  # New ...
            "pred_scores": pred_scores,
            "num_args": num_args,
            "num_preds": num_preds,
            "arg_labels": torch.max(srl_scores, 1)[1],  # [num_sentences, num_args, num_preds]
            "srl_scores": srl_scores
        })
        return predict_dict, srl_loss

    def save(self, filepath):
        """ Save model parameters to file.
        """
        torch.save(self.state_dict(), filepath)
        print('Saved model to: {}'.format(filepath))

    def load(self, filepath):
        """ Load model parameters from file.
        """
        model_params = torch.load(filepath)
        for model_param, pretrained_model_param in zip(self.parameters(), model_params.items()):
            if pretrained_model_param[1].size()[0] > 10000:  # pretrained word embedding
                pretrained_word_embedding_size = pretrained_model_param[1].size()[0]
                model_param.data[:pretrained_word_embedding_size].copy_(pretrained_model_param[1])
                print("Load {} pretrained word embedding!".format(pretrained_word_embedding_size))
            else:
                model_param.data.copy_(pretrained_model_param[1])
        print('Loaded model from: {}'.format(filepath))
