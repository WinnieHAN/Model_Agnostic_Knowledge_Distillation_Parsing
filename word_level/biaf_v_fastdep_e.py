from __future__ import print_function

__author__ = 'max'
"""
Implementation of Bi-directional LSTM-CNNs-TreeCRF model for Graph-based dependency parsing.
"""

import sys
import os, math, codecs

sys.path.append(".")
sys.path.append("..")

import time
import argparse
import uuid
import json

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, SGD, Adamax
from neuronlp2.io import get_logger, conllx_data
from neuronlp2.models import BiRecurrentConvBiAffine
from neuronlp2 import utils
from neuronlp2.io import CoNLLXWriter
from neuronlp2.tasks import parser
from neuronlp2.nn.utils import freeze_embedding
from seq2seq_rl.seq2seq import Seq2seq_Model
from seq2seq_rl.rl import LossRL, LossBiafRL, get_bleu, get_correct
# from stack_parser_eval import third_party_parser
from base_attack import glove_utils
from word_level.bridge_of_biaf_v import fastdep_predict
import tensorflow as tf
from dependency_parsing_tf.utils.feature_extraction import load_datasets, DataConfig
from dependency_parsing_tf.parser_model import ParserModel
from dependency_parsing_tf.utils.feature_extraction import Token, Sentence

uid = uuid.uuid4().hex[:6]

# 3 sub-models should be pretrained in our approach
#   seq2seq pretrain, denoising autoencoder  | or using token-wise adv to generate adv examples.
#   structure prediction model
#   oracle parser
# then we train the seq2seq model using rl


def main():
    args_parser = argparse.ArgumentParser(description='Tuning with graph-based parsing')
    args_parser.add_argument('--mode', choices=['RNN', 'LSTM', 'GRU', 'FastLSTM'], help='architecture of rnn', required=True)
    args_parser.add_argument('--cuda', action='store_true', help='using GPU')
    args_parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    args_parser.add_argument('--batch_size', type=int, default=64, help='Number of sentences in each batch')
    args_parser.add_argument('--hidden_size', type=int, default=256, help='Number of hidden units in RNN')
    args_parser.add_argument('--arc_space', type=int, default=128, help='Dimension of tag space')
    args_parser.add_argument('--type_space', type=int, default=128, help='Dimension of tag space')
    args_parser.add_argument('--num_layers', type=int, default=1, help='Number of layers of RNN')
    args_parser.add_argument('--num_filters', type=int, default=50, help='Number of filters in CNN')
    args_parser.add_argument('--pos', action='store_true', help='use part-of-speech embedding.')
    args_parser.add_argument('--char', action='store_true', help='use character embedding and CNN.')
    args_parser.add_argument('--pos_dim', type=int, default=50, help='Dimension of POS embeddings')
    args_parser.add_argument('--char_dim', type=int, default=50, help='Dimension of Character embeddings')
    args_parser.add_argument('--opt', choices=['adam', 'sgd', 'adamax'], help='optimization algorithm')
    args_parser.add_argument('--objective', choices=['cross_entropy', 'crf'], default='cross_entropy', help='objective function of training procedure.')
    args_parser.add_argument('--decode', choices=['mst', 'greedy'], help='decoding algorithm', required=True)
    args_parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    args_parser.add_argument('--decay_rate', type=float, default=0.05, help='Decay rate of learning rate')
    args_parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
    args_parser.add_argument('--gamma', type=float, default=0.0, help='weight for regularization')
    args_parser.add_argument('--epsilon', type=float, default=1e-8, help='epsilon for adam or adamax')
    args_parser.add_argument('--p_rnn', nargs=2, type=float, required=True, help='dropout rate for RNN')
    args_parser.add_argument('--p_in', type=float, default=0.33, help='dropout rate for input embeddings')
    args_parser.add_argument('--p_out', type=float, default=0.33, help='dropout rate for output layer')
    args_parser.add_argument('--schedule', type=int, help='schedule for learning rate decay')
    args_parser.add_argument('--unk_replace', type=float, default=0., help='The rate to replace a singleton word with UNK')
    args_parser.add_argument('--punctuation', nargs='+', type=str, help='List of punctuations')
    args_parser.add_argument('--word_embedding', choices=['glove', 'senna', 'sskip', 'polyglot'], help='Embedding for words', required=True)
    args_parser.add_argument('--word_path', help='path for word embedding dict')
    args_parser.add_argument('--freeze', action='store_true', help='frozen the word embedding (disable fine-tuning).')
    args_parser.add_argument('--char_embedding', choices=['random', 'polyglot'], help='Embedding for characters', required=True)
    args_parser.add_argument('--char_path', help='path for character embedding dict')
    args_parser.add_argument('--train')  # "data/POS-penn/wsj/split1/wsj1.train.original"
    args_parser.add_argument('--dev')  # "data/POS-penn/wsj/split1/wsj1.dev.original"
    args_parser.add_argument('--test')  # "data/POS-penn/wsj/split1/wsj1.test.original"
    args_parser.add_argument('--model_path', help='path for saving model file.', required=True)
    args_parser.add_argument('--model_name', help='name for saving model file.', required=True)

    args_parser.add_argument('--seq2seq_save_path', default='models/seq2seq/seq2seq_save_model', type=str, help='seq2seq_save_path')
    args_parser.add_argument('--network_save_path', default='models/seq2seq/network_save_model', type=str, help='network_save_path')

    args_parser.add_argument('--seq2seq_load_path', default='models/seq2seq/seq2seq_save_model', type=str, help='seq2seq_load_path')
    args_parser.add_argument('--network_load_path', default='models/seq2seq/network_save_model', type=str, help='network_load_path')

    args_parser.add_argument('--rl_finetune_seq2seq_save_path', default='models/rl_finetune/seq2seq_save_model', type=str, help='rl_finetune_seq2seq_save_path')
    args_parser.add_argument('--rl_finetune_network_save_path', default='models/rl_finetune/network_save_model', type=str, help='rl_finetune_network_save_path')

    args_parser.add_argument('--rl_finetune_seq2seq_load_path', default='models/rl_finetune/seq2seq_save_model', type=str, help='rl_finetune_seq2seq_load_path')
    args_parser.add_argument('--rl_finetune_network_load_path', default='models/rl_finetune/network_save_model', type=str, help='rl_finetune_network_load_path')

    args_parser.add_argument('--treebank', type=str, default='ctb', help='tree bank', choices=['ctb', 'ptb'])  # ctb

    args = args_parser.parse_args()

    # args.train = "data/ptb/dev.conllu"
    # args.dev = "data/ptb/dev.conllu"
    # args.test = "data/ptb/dev.conllu"

    logger = get_logger("GraphParser")
    # SEED = 0
    # torch.manual_seed(SEED)
    # torch.cuda.manual_seed(SEED)

    mode = args.mode
    obj = args.objective
    decoding = args.decode
    train_path = args.train
    dev_path = args.dev
    test_path = args.test
    model_path = args.model_path
    model_name = args.model_name
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    arc_space = args.arc_space
    type_space = args.type_space
    num_layers = args.num_layers
    num_filters = args.num_filters
    learning_rate = args.learning_rate
    opt = args.opt
    momentum = 0.9
    betas = (0.9, 0.9)
    eps = args.epsilon
    decay_rate = args.decay_rate
    clip = args.clip
    gamma = args.gamma
    schedule = args.schedule
    p_rnn = tuple(args.p_rnn)
    p_in = args.p_in
    p_out = args.p_out
    unk_replace = args.unk_replace
    punctuation = args.punctuation

    freeze = args.freeze
    word_embedding = args.word_embedding
    word_path = args.word_path

    use_char = args.char
    char_embedding = args.char_embedding
    char_path = args.char_path

    use_pos = args.pos
    pos_dim = args.pos_dim
    word_dict, word_dim = utils.load_embedding_dict(word_embedding, word_path)

    char_dict = None
    char_dim = args.char_dim
    if char_embedding != 'random':
        char_dict, char_dim = utils.load_embedding_dict(char_embedding, char_path)

    logger.info("Creating Alphabets")
    alphabet_path = os.path.join(model_path, 'alphabets/')
    model_name = os.path.join(model_path, model_name)
    word_alphabet, char_alphabet, pos_alphabet, type_alphabet = conllx_data.create_alphabets(alphabet_path, train_path, data_paths=[dev_path, test_path],
                                                                                             max_vocabulary_size=100000, embedd_dict=word_dict)

    num_words = word_alphabet.size()
    num_chars = char_alphabet.size()
    num_pos = pos_alphabet.size()
    num_types = type_alphabet.size()

    logger.info("Word Alphabet Size: %d" % num_words)
    logger.info("Character Alphabet Size: %d" % num_chars)
    logger.info("POS Alphabet Size: %d" % num_pos)
    logger.info("Type Alphabet Size: %d" % num_types)

    logger.info("Reading Data")
    device = torch.device('cuda:0')  #torch.device('cuda:0') if args.cuda else torch.device('cpu') #TODO:8.8

    data_train = conllx_data.read_data_to_tensor(train_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, symbolic_root=True, device=device)
    # data_train = conllx_data.read_data(train_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    # num_data = sum([len(bucket) for bucket in data_train])
    num_data = sum(data_train[1])

    data_dev = conllx_data.read_data_to_tensor(dev_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, symbolic_root=True, device=device)
    data_test = conllx_data.read_data_to_tensor(test_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, symbolic_root=True, device=device)

    punct_set = None
    if punctuation is not None:
        punct_set = set(punctuation)
        logger.info("punctuations(%d): %s" % (len(punct_set), ' '.join(punct_set)))

    def construct_word_embedding_table():
        scale = np.sqrt(3.0 / word_dim)
        table = np.empty([word_alphabet.size(), word_dim], dtype=np.float32)
        table[conllx_data.UNK_ID, :] = np.zeros([1, word_dim]).astype(np.float32) if freeze else np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)
        oov = 0
        for word, index in word_alphabet.items():
            if word in word_dict:
                embedding = word_dict[word]
            elif word.lower() in word_dict:
                embedding = word_dict[word.lower()]
            else:
                embedding = np.zeros([1, word_dim]).astype(np.float32) if freeze else np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)
                oov += 1
            table[index, :] = embedding
        print('word OOV: %d' % oov)
        return torch.from_numpy(table)

    def construct_char_embedding_table():
        if char_dict is None:
            return None

        scale = np.sqrt(3.0 / char_dim)
        table = np.empty([num_chars, char_dim], dtype=np.float32)
        table[conllx_data.UNK_ID, :] = np.random.uniform(-scale, scale, [1, char_dim]).astype(np.float32)
        oov = 0
        for char, index, in char_alphabet.items():
            if char in char_dict:
                embedding = char_dict[char]
            else:
                embedding = np.random.uniform(-scale, scale, [1, char_dim]).astype(np.float32)
                oov += 1
            table[index, :] = embedding
        print('character OOV: %d' % oov)
        return torch.from_numpy(table)

    word_table = construct_word_embedding_table()
    char_table = construct_char_embedding_table()

    # Pretrain structure prediction model (biaff model). model name: network
    window = 3
    if obj == 'cross_entropy':
        network = BiRecurrentConvBiAffine(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, num_filters, window,
                                          mode, hidden_size, num_layers, num_types, arc_space, type_space,
                                          embedd_word=word_table, embedd_char=char_table,
                                          p_in=p_in, p_out=p_out, p_rnn=p_rnn, biaffine=True, pos=use_pos, char=use_char)
    elif obj == 'crf':
        raise NotImplementedError
    else:
        raise RuntimeError('Unknown objective: %s' % obj)

    def save_args():
        arg_path = model_name + '.arg.json'
        arguments = [word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, num_filters, window,
                     mode, hidden_size, num_layers, num_types, arc_space, type_space]
        kwargs = {'p_in': p_in, 'p_out': p_out, 'p_rnn': p_rnn, 'biaffine': True, 'pos': use_pos, 'char': use_char}
        json.dump({'args': arguments, 'kwargs': kwargs}, open(arg_path, 'w'), indent=4)

    if freeze:
        freeze_embedding(network.word_embedd)

    network = network.to(device)

    save_args()

    pred_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    gold_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)

    def generate_optimizer(opt, lr, params):
        params = filter(lambda param: param.requires_grad, params)
        if opt == 'adam':
            return Adam(params, lr=lr, betas=betas, weight_decay=gamma, eps=eps)
        elif opt == 'sgd':
            return SGD(params, lr=lr, momentum=momentum, weight_decay=gamma, nesterov=True)
        elif opt == 'adamax':
            return Adamax(params, lr=lr, betas=betas, weight_decay=gamma, eps=eps)
        else:
            raise ValueError('Unknown optimization algorithm: %s' % opt)

    lr = learning_rate
    optim = generate_optimizer(opt, lr, network.parameters())
    opt_info = 'opt: %s, ' % opt
    if opt == 'adam':
        opt_info += 'betas=%s, eps=%.1e' % (betas, eps)
    elif opt == 'sgd':
        opt_info += 'momentum=%.2f' % momentum
    elif opt == 'adamax':
        opt_info += 'betas=%s, eps=%.1e' % (betas, eps)

    word_status = 'frozen' if freeze else 'fine tune'
    char_status = 'enabled' if use_char else 'disabled'
    pos_status = 'enabled' if use_pos else 'disabled'
    logger.info("Embedding dim: word=%d (%s), char=%d (%s), pos=%d (%s)" % (word_dim, word_status, char_dim, char_status, pos_dim, pos_status))
    logger.info("CNN: filter=%d, kernel=%d" % (num_filters, window))
    logger.info("RNN: %s, num_layer=%d, hidden=%d, arc_space=%d, type_space=%d" % (mode, num_layers, hidden_size, arc_space, type_space))
    logger.info("train: obj: %s, l2: %f, (#data: %d, batch: %d, clip: %.2f, unk replace: %.2f)" % (obj, gamma, num_data, batch_size, clip, unk_replace))
    logger.info("dropout(in, out, rnn): (%.2f, %.2f, %s)" % (p_in, p_out, p_rnn))
    logger.info("decoding algorithm: %s" % decoding)
    logger.info(opt_info)


    if decoding == 'greedy':
        decode = network.decode
    elif decoding == 'mst':
        decode = network.decode_mst
    else:
        raise ValueError('Unknown decoding algorithm: %s' % decoding)

    print('Pretrain biaffine model.')
    patient = 0
    decay = 0
    max_decay = 9
    double_schedule_decay = 5
    num_epochs = 1  # debug hanwj
    if args.treebank == 'ptb':
        network.load_state_dict(torch.load('models/parsing/biaffine/network.pt'))  # TODO: 10.7
    elif args.treebank == 'ctb':
        network.load_state_dict(torch.load('ctb_models/parsing/biaffine/network.pt'))  # TODO: 10.7
    # network.load_state_dict(torch.load('models/parsing/biaffine/network.pt'))  # TODO: 7.13
    network.to(device)
    for epoch in range(1, num_epochs + 1):
        print('Epoch %d (%s, optim: %s, learning rate=%.6f, eps=%.1e, decay rate=%.2f (schedule=%d, patient=%d, decay=%d)): ' % (epoch, mode, opt, lr, eps, decay_rate, schedule, patient, decay))
        train_err = 0.
        train_err_arc = 0.
        train_err_type = 0.
        train_total = 0.
        start_time = time.time()
        num_back = 0
        network.train()
        FLAG = True
        for para in network.parameters():
            if FLAG:
                para.requires_grad = True
                FLAG = False
            else:
                para.requires_grad = False
        # network.state_dict().items()[0][1].requires_grad = True
        # ori_word_embedding = network.parameters().next().detach().data.cpu().numpy().T  # torch.Size([35374, 100])
        # c_ = -2 * np.dot(ori_word_embedding.T, ori_word_embedding)
        # a = np.sum(np.square(ori_word_embedding), axis=0).reshape((1, -1))
        # b = a.T
        # dist = a + b + c_
        # np.save('base_attach/dist_counter_35374.npy', dist)
        def dataform_biaf2fastdep(batch):
            # word, char, pos, heads, types, masks, lengths = batch
            wordss, charss, poss, headss, typess, maskss, lengthss = batch
            fastdep_batch = []
            for bi in range(len(wordss)):
                tokens = []
                for wi in range(1, lengthss[bi]):  # there is ROOT in the first token.
                    token_index = wi - 1
                    word = word_alphabet.get_instance(wordss[bi][wi])
                    pos = pos_alphabet.get_instance(poss[bi][wi])
                    dep = type_alphabet.get_instance(typess[bi][wi])
                    head_index = headss[bi][wi] - 1
                    token = Token(token_index, word, pos, dep, head_index)
                    tokens.append(token)
                sentence = Sentence(tokens)
                fastdep_batch.append(sentence)
            return fastdep_batch

        outf = 'word_level/adv_sentences025.txt'
        wf = codecs.open(outf, 'w', encoding='utf8')
        outf_conllu = 'word_level/adv_sentences025.conllu'
        wf_conllu = codecs.open(outf_conllu, 'w', encoding='utf8')
        ori_outf = 'word_level/ori_sentences.txt'
        ori_wf = codecs.open(ori_outf, 'w', encoding='utf8')
        ori_word_embedding = network.parameters().next().detach().data.cpu().numpy().T  # torch.Size([35374, 100])
        kk = 0
        print('batch_size: ', str(batch_size))
        with tf.Graph().as_default(), tf.Session() as sess:
            with tf.variable_scope("model", reuse=tf.AUTO_REUSE) as model_scope:  # reuse=tf.AUTO_REUSE or reuse=True
                dataset = load_datasets(load_existing_dump=True)
                config = dataset.model_config
                model = ParserModel(config, dataset.word_embedding_matrix, dataset.pos_embedding_matrix,
                                    dataset.dep_embedding_matrix)
                saver = tf.train.Saver()
                model_dir = 'params_2020-01-28'
                ckpt_path = tf.train.latest_checkpoint(os.path.join(DataConfig.data_dir_path, model_dir))
                saver.restore(sess, ckpt_path)
                for batch in conllx_data.iterate_batch_tensor(data_train, batch_size):  # TODO: data_train
                    kk = kk + 1
                    print('--------'+str(kk)+'--------')
                    test_data = dataform_biaf2fastdep(batch)
                    seudo_heads = fastdep_predict(test_data=test_data, dataset=dataset, sess=sess, model=model, batch_max_length=batch[0].shape[1])

                    #----------------
                    optim.zero_grad()
                    seudo_heads = torch.from_numpy(np.array(seudo_heads)).long().to(device)
                    word, char, pos, heads, types, masks, lengths = batch
                    loss_arc, loss_type = network.loss(word, char, pos, seudo_heads, types, mask=masks, length=lengths)  # head should be replaced with seudo_heads
                    loss = loss_arc  #+  loss_type
                    loss = - loss
                    loss.backward()
                    clip_grad_norm_(network.parameters(), clip)
                    optim.step()

                    for batch_i in range(len(word)):
                        ori_wf.write(' '.join([str(word_alphabet.get_instance(wordi)) for wordi in word[batch_i][1:] if not wordi==1]).encode('utf-8'))
                        ori_wf.write('\n')
                    # adv_word_embedding = network.parameters().next().detach().data.cpu().numpy().T  #torch.Size([35374, 100])
                    for batch_i in range(len(word)):
                        changed_num = 0
                        for batch_j in range(1, lengths[batch_i].cpu().numpy()-1):
                            one_word = word[batch_i][batch_j].item()
                            if one_word==1:
                                continue
                            src_word_idx = one_word
                            adv_vector = network.parameters().next().detach().data.cpu().numpy()[src_word_idx,:]
                            ori_word_embedding[:, src_word_idx] = adv_vector
                            c_ = -2 * np.dot(ori_word_embedding.T, adv_vector)
                            a = np.sum(np.square(ori_word_embedding), axis=0)#.reshape((1, -1))
                            b = np.sum(np.square(adv_vector))
                            dist = a + c_ + b
                            neighbours, _ = glove_utils.pick_most_similar_words_from_vector(src_word_idx, adv_vector, dist, ret_count=3, threshold=20)
                            if word[batch_i][batch_j] == neighbours[0]:
                                continue
                            if changed_num*1.0/(lengths[batch_i].cpu().numpy()-1)>0.1:
                                break
                            word[batch_i][batch_j] = neighbours[0]
                            changed_num = changed_num + 1

                    for batch_i in range(len(word)):
                        wf.write(' '.join([str(word_alphabet.get_instance(wordi)) for wordi in word[batch_i][1:] if not wordi==1]))
                        wf.write('\n')
                    for batch_i in range(len(word)):
                        ws = [str(word_alphabet.get_instance(wordi)) for wordi in word[batch_i][1:] if not wordi==1]
                        for w_i in range(len(ws)):
                            wf_conllu.write(str(w_i+1) + '\t'+ws[w_i] + '\t'+'_' + '\t'+'NNP'+ '\t'+ 'NNP'+'\t' +'_'+'\t'+str(1)+ '\tnn\t_\t_\n')
                        wf_conllu.write('\n')
        wf.close()
        ori_wf.close()
        wf_conllu.close()

    print('Baseline Attack.')


if __name__ == '__main__':
    main()
