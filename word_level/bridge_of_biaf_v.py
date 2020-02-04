from dependency_parsing_tf.parser_model import ParserModel

import os
import time
import tensorflow as tf
from dependency_parsing_tf.utils.feature_extraction import load_datasets, DataConfig, Flags, punc_pos, pos_prefix
from dependency_parsing_tf.utils.tf_utils import visualize_sample_embeddings

def highlight_string(temp):
    print 80 * "="
    print temp
    print 80 * "="


def main(flag, load_existing_dump=False):
    print "loading data.."

    dataset = load_datasets(load_existing_dump)
    config = dataset.model_config

    print "word vocab Size: {}".format(len(dataset.word2idx))
    print "pos vocab Size: {}".format(len(dataset.pos2idx))
    print "dep vocab Size: {}".format(len(dataset.dep2idx))
    print "Training Size: {}".format(len(dataset.train_inputs[0]))
    print "valid data Size: {}".format(len(dataset.valid_data))
    print "test data Size: {}".format(len(dataset.test_data))

    print len(dataset.word2idx), len(dataset.word_embedding_matrix)
    print len(dataset.pos2idx), len(dataset.pos_embedding_matrix)
    print len(dataset.dep2idx), len(dataset.dep_embedding_matrix)

    if not os.path.exists(os.path.join(DataConfig.data_dir_path, DataConfig.model_dir)):
        os.makedirs(os.path.join(DataConfig.data_dir_path, DataConfig.model_dir))

    with tf.Graph().as_default(), tf.Session() as sess:
        print "Building network...",
        start = time.time()
        with tf.variable_scope("model") as model_scope:
            model = ParserModel(config, dataset.word_embedding_matrix, dataset.pos_embedding_matrix,
                                dataset.dep_embedding_matrix)
            saver = tf.train.Saver()
            """
            model_scope.reuse_variables()
                -> no need to call tf.variable_scope(model_scope, reuse = True) again
                -> directly access variables & call functions inside this block itself.
                -> ref: https://www.tensorflow.org/versions/r1.2/api_docs/python/tf/variable_scope
                -> https://stackoverflow.com/questions/35919020/whats-the-difference-of-name-scope-and-a-variable-scope-in-tensorflow
            """

        print "took {:.2f} seconds\n".format(time.time() - start)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(DataConfig.data_dir_path, DataConfig.summary_dir,
                                                          DataConfig.train_summ_dir), sess.graph)
        valid_writer = tf.summary.FileWriter(os.path.join(DataConfig.data_dir_path, DataConfig.summary_dir,
                                                          DataConfig.test_summ_dir))

        if flag == Flags.TRAIN:

            # Variable initialization -> not needed for .restore()
            """ The variables to restore do not have to have been initialized,
            as restoring is itself a way to initialize variables. """
            sess.run(tf.global_variables_initializer())
            """ call 'assignment' after 'init' only, else 'assignment' will get reset by 'init' """
            sess.run(tf.assign(model.word_embedding_matrix, model.word_embeddings))
            sess.run(tf.assign(model.pos_embedding_matrix, model.pos_embeddings))
            sess.run(tf.assign(model.dep_embedding_matrix, model.dep_embeddings))

            highlight_string("TRAINING")
            model.print_trainable_varibles()

            model.fit(sess, saver, config, dataset, train_writer, valid_writer, merged)

            # Testing
            highlight_string("Testing")
            print "Restoring best found parameters on dev set"
            saver.restore(sess, os.path.join(DataConfig.data_dir_path, DataConfig.model_dir,
                                             DataConfig.model_name))
            model.compute_dependencies(sess, dataset.test_data, dataset)
            test_UAS = model.get_UAS(dataset.test_data)
            print "test UAS: {}".format(test_UAS * 100)

            train_writer.close()
            valid_writer.close()

            # visualize trained embeddings after complete training (not after each epoch)
            with tf.variable_scope(model_scope, reuse=True):
                pos_emb = tf.get_variable("feature_lookup/pos_embedding_matrix",
                                          [len(dataset.pos2idx.keys()), dataset.model_config.embedding_dim])
                visualize_sample_embeddings(sess, os.path.join(DataConfig.data_dir_path, DataConfig.model_dir),
                                            dataset.pos2idx.keys(), dataset.pos2idx, pos_emb)
            print "to Visualize Embeddings, run in terminal:"
            print "tensorboard --logdir=" + os.path.abspath(os.path.join(DataConfig.data_dir_path,
                                                                         DataConfig.model_dir))

        else:
            ckpt_path = tf.train.latest_checkpoint(os.path.join(DataConfig.data_dir_path,
                                                                DataConfig.model_dir))
            if ckpt_path is not None:
                print "Found checkpoint! Restoring variables.."
                saver.restore(sess, ckpt_path)
                highlight_string("Testing")
                model.compute_dependencies(sess, dataset.test_data, dataset)
                test_UAS = model.get_UAS(dataset.test_data)
                print "test UAS: {}".format(test_UAS * 100)
                # model.run_valid_epoch(sess, dataset.valid_data, dataset)
                # valid_UAS = model.get_UAS(dataset.valid_data)
                # print "valid UAS: {}".format(valid_UAS * 100)

                highlight_string("Embedding Visualization")
                with tf.variable_scope(model_scope, reuse=True):
                    pos_emb = tf.get_variable("feature_lookup/pos_embedding_matrix",
                                              [len(dataset.pos2idx.keys()), dataset.model_config.embedding_dim])
                    visualize_sample_embeddings(sess, os.path.join(DataConfig.data_dir_path, DataConfig.model_dir),
                                                dataset.pos2idx.keys(), dataset.pos2idx, pos_emb)
                print "to Visualize Embeddings, run in terminal:"
                print "tensorboard --logdir=" + os.path.abspath(os.path.join(DataConfig.data_dir_path,
                                                                             DataConfig.model_dir))

            else:
                print "No checkpoint found!"

def fastdep_predict(test_data, dataset, sess, model, batch_max_length):

    # print "word vocab Size: {}".format(len(dataset.word2idx))

    model.compute_dependencies(sess, test_data, dataset)
    test_UAS = model.get_UAS(test_data)
    # print "test UAS: {}".format(test_UAS * 100)
    seudo_heads = []
    # test_data = dataset.test_data
    for i in range(len(test_data)):
        sentence = test_data[i]
        stc_max_length = len(sentence.tokens)
        head = [-2 for _ in range(stc_max_length)]
        for h, t, in sentence.predicted_dependencies:
            head[t.token_id] = h.token_id
        head = [0] + [i+1 for i in head] + [0 for _ in range(batch_max_length-stc_max_length-1)]
        seudo_heads = seudo_heads + [head]
    return seudo_heads
    # model.run_valid_epoch(sess, dataset.valid_data, dataset)
    # valid_UAS = model.get_UAS(dataset.valid_data)
    # print "valid UAS: {}".format(valid_UAS * 100)





if __name__ == '__main__':
    # fastdep_predict(Flags.TEST, load_existing_dump=True)
    dataset = load_datasets(load_existing_dump=True)
    with tf.Graph().as_default(), tf.Session() as sess:
        with tf.variable_scope("model") as model_scope:
            config = dataset.model_config
            model = ParserModel(config, dataset.word_embedding_matrix, dataset.pos_embedding_matrix,
                                dataset.dep_embedding_matrix)
            saver = tf.train.Saver()
            """
            model_scope.reuse_variables()
                -> no need to call tf.variable_scope(model_scope, reuse = True) again
                -> directly access variables & call functions inside this block itself.
                -> ref: https://www.tensorflow.org/versions/r1.2/api_docs/python/tf/variable_scope
                -> https://stackoverflow.com/questions/35919020/whats-the-difference-of-name-scope-and-a-variable-scope-in-tensorflow
            """
            ckpt_path = tf.train.latest_checkpoint(os.path.join(DataConfig.data_dir_path, DataConfig.model_dir))
            print "Found checkpoint! Restoring variables.."
            saver.restore(sess, ckpt_path)
            seudo_heads = fastdep_predict(test_data=dataset.test_data, dataset=dataset, sess=sess, model=model)