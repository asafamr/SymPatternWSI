from .bilm_interface import Bilm
from spwsi.semeval_utils import generate_sem_eval_2013, evaluate_labeling
from collections import defaultdict
from .wsi_clustering import cluster_inst_ids_representatives
from tqdm import tqdm
import logging
import os
import numpy as np

DEFAULT_PARAMS = dict(
    n_clusters=7,
    n_represent=20,
    n_samples_side=4,
    cuda_device=0,
    debug_dir='debug',
    disable_lemmatization=False,
    disable_symmetric_patterns=False,
    disable_tfidf=False,
    run_postfix='',
    lm_batch_size=50,
    prediction_cutoff=50,
    cutoff_lm_vocab=50000,
)


class SPWSI:
    def __init__(self, bilm: Bilm):
        self.bilm = bilm

    def run(self, n_clusters, n_represent, n_samples_side, disable_tfidf, debug_dir, run_name,
            disable_symmetric_patterns, disable_lemmatization, prediction_cutoff,
            print_progress=False):

        semeval_dataset_by_target = defaultdict(dict)

        # SemEval target might be, for example, book.n (lemma+POS)
        # SemEval instance might be, for example, book.n.12 (target+index).
        # In the example instance above, corresponds to one usage of book as a noun in a sentence

        # semeval_dataset_by_target is a dict from target to dicts of instances with their sentence
        # so semeval_dataset_by_target['book.n']['book.n.12'] is the sentence tokens of the 'book.n.12' instance
        # and the index of book in these tokens

        # load all dataset to memory
        for tokens, target_idx, inst_id in generate_sem_eval_2013('./resources/SemEval-2013-Task-13-test-data'):
            lemma_pos = inst_id.rsplit('.', 1)[0]
            semeval_dataset_by_target[lemma_pos][inst_id] = (tokens, target_idx)

        inst_id_to_sense = {}
        gen = semeval_dataset_by_target.items()
        if print_progress:
            gen = tqdm(gen, desc='predicting substitutes')
        for lemma_pos, inst_id_to_sentence in gen:
            inst_ids_to_representatives = self.bilm.predict_sent_substitute_representatives(
                inst_id_to_sentence, n_represent, n_samples_side, disable_symmetric_patterns, disable_lemmatization,
                prediction_cutoff)
            clusters = cluster_inst_ids_representatives(inst_ids_to_representatives, n_clusters, disable_tfidf)
            inst_id_to_sense.update(clusters)

        out_key_path = None
        if debug_dir:
            out_key_path = os.path.join(debug_dir, run_name + '.key')
        scores = evaluate_labeling('./resources/SemEval-2013-Task-13-test-data', inst_id_to_sense, out_key_path)
        if print_progress:
            print('written SemEval key file to %s' % out_key_path)
        fnmi = scores['all']['FNMI']
        fbc = scores['all']['FBC']
        msg = 'results FNMI %.2f FBC %.2f AVG %.2f' % (fnmi * 100, fbc * 100, np.sqrt(fnmi * fbc) * 100)
        logging.info(msg)
        if print_progress:
            print(msg)
        return scores
