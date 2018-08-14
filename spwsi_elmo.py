from spwsi.semeval_utils import generate_sem_eval_2013, evaluate_labeling
from spwsi.bilm_elmo import BilmElmo
from spwsi.wsi_clustering import WsiClusterer
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import argparse
import os
import logging
from time import strftime

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BiLM Symmetric Patterns WSI Demo')
    parser.add_argument('--n-clusters', dest='n_clusters', type=int, default=8,
                        help='number of clusters per instance')
    parser.add_argument('--n-representatives', dest='n_represent', type=int, default=10,
                        help='number of representations per sentence')
    parser.add_argument('--n-samples-side', dest='n_samples_side', type=int, default=6,
                        help='number of samples per representations side')
    parser.add_argument('--cuda', type=int, default=0,
                        help='cuda device for ELMo (-1 to disable)')
    parser.add_argument('--debug-dir', dest='debug_dir', type=str, default='debug',
                        help='logs and keys are written will be written to this dir')
    parser.add_argument('--disable-lemmatization', dest='disable_lemmatization',
                        default=False, action='store_true',
                        help='disable ELMO prediction lemmatization')
    parser.add_argument('--disable-symmetric-patterns', dest='disable_symmetric_patterns',
                        default=False, action='store_true',
                        help='disable "x and y" symmetric pattern and predict substitutes inplace')
    parser.add_argument('--disable-tfidf', dest='disable_tfidf', action='store_true',
                        help='disable tfidf transformer')
    parser.add_argument('--append-results-file', dest='append_results_file', type=str, default=None,
                        help='append final run results to this file')
    parser.add_argument('--run-prefix', dest='run_prefix', type=str, default='',
                        help='will be prepended to log file names')
    parser.add_argument('--elmo-batch-size', dest='elmo_batch_size', type=int, default=40,
                        help='ELMo prediction batch size')
    parser.add_argument('--prediction-cutoff', dest='prediction_cutoff', type=int, default=50,
                        help='ELMo predicted distribution top K cutoff')
    parser.add_argument('--cutoff-elmo-vocab', dest='cutoff_elmo_vocab', type=int, default=100000,
                        help='optimization: only use top K words for faster output matrix multiplication')
    args = parser.parse_args()

    startmsg = 'BiLM Symmetric Patterns WSI Demo\n\n'
    startmsg += 'Arguments:\n'
    startmsg += '-' * 10 + '\n'
    for arg in vars(args):
        startmsg += (' %-30s:%s\n' % (arg.replace('_', '-'), getattr(args, arg)))
    startmsg = startmsg.strip()
    print(startmsg)

    run_name = args.run_prefix + '-' if args.run_prefix else ''
    run_name += strftime("%y-%m-%d-%H-%M-%S")
    if args.debug_dir:
        if not os.path.exists(args.debug_dir):
            os.makedirs(args.debug_dir)
        logging.basicConfig(filename=os.path.join(args.debug_dir, '%s.log.txt' % run_name),
                            format='%(asctime)s %(message)s', datefmt='%H:%M:%S',
                            level=logging.INFO)
    logging.info(startmsg)

    sentences_by_lemma_pos = defaultdict(dict)
    for tokens, target_idx, inst_id in generate_sem_eval_2013('./resources/SemEval-2013-Task-13-test-data'):
        lemma_pos = inst_id.rsplit('.', 1)[0]
        sentences_by_lemma_pos[lemma_pos][inst_id] = (tokens, target_idx)

    elmo = BilmElmo(args.cuda, './resources/elmo_2x4096_512_2048cnn_2xhighway_softmax_weights.hdf5',
                    './resources/vocab-2016-09-10.txt', lemmatize_predictions=not args.disable_lemmatization,
                    batch_size=args.elmo_batch_size, cutoff=args.prediction_cutoff,
                    cutoff_elmo_vocab=args.cutoff_elmo_vocab,
                    disable_symmetric_patterns=args.disable_symmetric_patterns)
    clusterer = WsiClusterer(elmo, args.n_represent, args.n_samples_side, args.disable_tfidf, args.n_clusters)
    inst_id_to_sense = {}
    for lemma_pos, inst_id_to_sentence in tqdm(sentences_by_lemma_pos.items(), desc='predicting substitutes'):
        inst_id_to_sense.update(clusterer.soft_cluster_sentences_to_senses(inst_id_to_sentence))

    out_key_path = None
    if args.debug_dir:
        out_key_path = os.path.join(args.debug_dir, run_name + '.key')
    fnmi, fbc = evaluate_labeling('./resources/SemEval-2013-Task-13-test-data', inst_id_to_sense, out_key_path)
    msg = 'results FNMI %.2f FBC %.2f AVG %.2f' % (fnmi * 100, fbc * 100, np.sqrt(fnmi * fbc) * 100)
    logging.info(msg)
    print(msg)
    if args.append_results_file:
        with open(args.append_results_file, 'a') as fout:
            fout.write('%s\t%.2f\t%.2f\t%.2f\n' % (run_name, fnmi * 100, fbc * 100, np.sqrt(fnmi * fbc) * 100))
