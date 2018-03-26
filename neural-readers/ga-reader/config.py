# Minibatch Size
BATCH_SIZE = 32
# Gradient clip threshold
GRAD_CLIP = 10
# Learning rate
LEARNING_RATE = 0.0005
#LEARNING_RATE = 0.
# Maximum number of steps in BPTT
GRAD_STEPS = -1
# Number of epochs for training
NUM_EPOCHS = 10
# do validation every VALIDATION_FREQ iterations
VALIDATION_FREQ = 100
# maximum word length for character model
MAX_WORD_LEN = 10

# dataset params
def get_params(dataset):
    if dataset=='cbtcn':
        return cbtcn_params
    elif dataset=='wdw' or dataset=='wdw_relaxed':
        return wdw_params
    elif dataset=='cnn':
        return cnn_params
    elif dataset=='dailymail':
        return dailymail_params
    elif dataset=='cbtne':
        return cbtne_params
    elif dataset=='clicr':
        return clicr_params
    elif dataset=='clicr_plain':
        return clicr_plain_params
    elif dataset=='clicr_novice':
        return clicr_novice_params
    else:
        raise ValueError("Dataset %s not found"%dataset)

cbtcn_params = {
        'nhidden'   :   128,
        'char_dim'  :   25,
        'dropout'   :   0.4,
        'word2vec'  :   'data/word2vec_glove.txt',
        'train_emb' :   0,
        'use_feat'  :   1,
        }

wdw_params = {
    'data_path' : "data/wdw/",
    'nhidden'   :   128,
    'char_dim'  :   0,#25,
    'dropout'   :   0.3,
    'word2vec'  :   'data/word2vec_glove.txt',
    'train_emb' :   0,
    'use_feat'  :   1,
        }

clicr_novice_params = {
    # THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python3 run.py --dataset clicr_novice --mode 1 --ent_setup ent-anonym --experiments_path ../experiments/ --data_path data/
    'test_file': '/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/bmj_case_reports_data/dataset_json_concept_annotated/test1.0.json',
    'validation_file': '/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/bmj_case_reports_data/dataset_json_concept_annotated/dev1.0.json',
    'nhidden': 128,
    'char_dim': 0,
    'dropout': 0.2,
    'word2vec': '/nas/corpora/accumulate/clicr/embeddings/005853d6-7164-11e7-b58e-901b0e5592c8/embeddings_with_header', # NYT+Wikipedia, 200d
    'train_emb': 1,
    'use_feat': 0
}

clicr_params = {
    'data_path' : "/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/bmj_case_reports_data/dataset_json_concept_annotated/",
    'nhidden': 128,
    'char_dim': 0,
    'dropout': 0.2,
    'word2vec': '/nas/corpora/accumulate/clicr/embeddings/de004a58-6eef-11e7-ac2f-901b0e5592c8/embeddings', # Pubmed, 200d
    'train_emb': 1,
    'use_feat': 0,
    'test_file': '/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/bmj_case_reports_data/dataset_json_concept_annotated/test1.0.json',
    #'test_file': 'data/test1.0.json',
    'validation_file': '/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/bmj_case_reports_data/dataset_json_concept_annotated/dev1.0.json'
    #'validation_file': 'data/dev1.0.json'
}

clicr_plain_params = {
    'data_path': "dataset_plain/no-ent/",
    'test_file' : '/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/bmj_case_reports_data/dataset_json_concept_annotated/test1.0.json',
    'validation_file': '/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/bmj_case_reports_data/dataset_json_concept_annotated/dev1.0.json',
    'nhidden': 128,
    'char_dim': 0,
    'dropout': 0.2,
    'word2vec': '/nas/corpora/accumulate/clicr/embeddings/de004a58-6eef-11e7-ac2f-901b0e5592c8/embeddings', # Pubmed, 200d
    'train_emb': 1,
    'use_feat': 0
}

cnn_params = {
    'data_path': "data/CNN_DailyMail/cnn/questions/",
    #'nhidden'   :   256,
    'nhidden'   :   128,
    'char_dim'  :   0,
    'dropout'   :   0.2,
    'word2vec'  :   '/nas/corpora/accumulate/clicr/embeddings/005853d6-7164-11e7-b58e-901b0e5592c8/embeddings_with_header', # NYT+Wikipedia, 200d
    'train_emb' :   1,
    'use_feat'  :   0,
    }

dailymail_params = {
        'nhidden'   :   256,
        'char_dim'  :   0,
        'dropout'   :   0.1,
        'word2vec'  :   'data/word2vec_glove.txt',
        'train_emb' :   1,
        'use_feat'  :   0,
        }

cbtne_params = {
        'nhidden'   :   128,
        'char_dim'  :   25,
        'dropout'   :   0.4,
        'word2vec'  :   'data/word2vec_glove.txt',
        'train_emb' :   0,
        'use_feat'  :   1,
        }
