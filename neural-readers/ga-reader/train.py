import os
import shutil
import time

from config import *
from model import GAReader
from utils import Helpers, DataPreprocessor, MiniBatchLoader


def main(save_path, params):
    nhidden = params['nhidden']
    dropout = params['dropout']
    word2vec = params['word2vec']
    dataset = params['dataset']
    nlayers = params['nlayers']
    train_emb = params['train_emb']
    char_dim = params['char_dim']
    use_feat = params['use_feat']
    gating_fn = params['gating_fn']
    ent_setup = params['ent_setup']  # ent, ent-anonym, no-ent
    data_path = params['data_path']
    # save settings
    shutil.copyfile('config.py', '%s/config.py' % save_path)

    use_chars = char_dim > 0

    if dataset == "clicr":
        dp = DataPreprocessor.DataPreprocessorClicr()
        data = dp.preprocess(
            #"/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/bmj_case_reports_data/dataset_json_concept_annotated/",
            data_path, ent_setup=ent_setup, no_training_set=False, use_chars=use_chars)
    elif dataset == "clicr_novice":
        dp = DataPreprocessor.DataPreprocessorNovice()
        data = dp.preprocess(
            data_path, ent_setup=ent_setup, no_training_set=False, use_chars=use_chars)
    else:
        dp = DataPreprocessor.DataPreprocessor()
        data = dp.preprocess(data_path, no_training_set=False, use_chars=use_chars)

    print("building minibatch loaders ...")
    batch_loader_train = MiniBatchLoader.MiniBatchLoader(data.training, BATCH_SIZE,
                                                         sample=1.0)
    batch_loader_val = MiniBatchLoader.MiniBatchLoader(data.validation, BATCH_SIZE)

    print("building network ...")
    W_init, embed_dim, = Helpers.load_word2vec_embeddings(data.dictionary[0], word2vec)
    m = GAReader.Model(nlayers, data.vocab_size, data.num_chars, W_init,
                       nhidden, embed_dim, dropout, train_emb,
                       char_dim, use_feat, gating_fn)

    print("training ...")
    num_iter = 0
    max_acc = 0.
    deltas = []

    logger = open(save_path + '/log', 'a')

    if os.path.isfile('%s/best_model.p' % save_path):
        print('loading previously saved model')
        m.load_model('%s/best_model.p' % save_path)
    else:
        print('saving init model')
        m.save_model('%s/model_init.p' % save_path)
        print('loading init model')
        m.load_model('%s/model_init.p' % save_path)

    for epoch in range(NUM_EPOCHS):
        estart = time.time()
        new_max = False

        for dw, dt, qw, qt, a, m_dw, m_qw, tt, tm, c, m_c, cl, fnames in batch_loader_train:
            loss, tr_acc, probs = m.train(dw, dt, qw, qt, c, a, m_dw, m_qw, tt, tm, m_c, cl)

            message = "Epoch %d TRAIN loss=%.4e acc=%.4f elapsed=%.1f" % (
                epoch, loss, tr_acc, time.time() - estart)
            print(message)
            logger.write(message + '\n')

            num_iter += 1
            if num_iter % VALIDATION_FREQ == 0:
                total_loss, total_acc, n, n_cand = 0., 0., 0, 0.

                for dw, dt, qw, qt, a, m_dw, m_qw, tt, tm, c, m_c, cl, fnames in batch_loader_val:
                    outs = m.validate(dw, dt, qw, qt, c, a,
                                      m_dw, m_qw, tt, tm, m_c, cl)
                    loss, acc, probs = outs[:3]

                    bsize = dw.shape[0]
                    total_loss += bsize * loss
                    total_acc += bsize * acc
                    n += bsize
                val_acc = total_acc / n
                if val_acc > max_acc:
                    max_acc = val_acc
                    m.save_model('%s/best_model.p' % save_path)
                    new_max = True
                message = "Epoch %d VAL loss=%.4e acc=%.4f max_acc=%.4f" % (
                    epoch, total_loss / n, val_acc, max_acc)
                print(message)
                logger.write(message + '\n')

        # m.save_model('%s/model_%d.p'%(save_path,epoch))
        message = "After Epoch %d: Train acc=%.4f, Val acc=%.4f" % (epoch, tr_acc, val_acc)
        print(message)
        logger.write(message + '\n')

        # learning schedule
        if epoch >= 2:
            m.anneal()
        # stopping criterion
        if not new_max:
            break

    logger.close()
