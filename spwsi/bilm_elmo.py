from spwsi.bilm_interface import Bilm
from allennlp.commands.elmo import ElmoEmbedder
from typing import List, Dict, Tuple
import multiprocessing
import numpy as np
from tqdm import tqdm
import os.path
import h5py
import logging


class BilmElmo(Bilm):

    def __init__(self, cuda_device, weights_path, vocab_path, lemmatize_predictions, batch_size=40, cutoff=50,
                 cutoff_elmo_vocab=50000, disable_symmetric_patterns=False):
        super().__init__()
        logging.info(
            'creating elmo in device %d. weight path %s, vocab_path %s, lemmatize_predictions %s,'
            ' batch_size: %d disable_symmetric_patterns:%s' % (
                cuda_device, weights_path, vocab_path, lemmatize_predictions,
                batch_size, disable_symmetric_patterns))
        self.elmo = ElmoEmbedder(cuda_device=cuda_device)

        self.batch_size = batch_size
        self.cutoff = cutoff
        self.disable_symmetric_patterns = disable_symmetric_patterns
        logging.info('warming up elmo')
        self._warm_up_elmo()
        logging.info('reading elmo weights')
        with h5py.File(weights_path) as fin:
            self.elmo_softmax_w = fin['softmax/W'][:].transpose()
        self.elmo_word_vocab = []

        def add_words_from_lines(lines):
            self.elmo_word_vocab = []
            stop_words = {'<UNK>', '<S>', '</S>', '--', '..', '...', '....'}
            rows_delete = []
            for idx, line in enumerate(lines):
                word = line.strip()
                if word in stop_words or len(word) <= 1:
                    rows_delete.append(idx)
                    continue
                self.elmo_word_vocab.append(word)
            self.elmo_softmax_w = np.delete(self.elmo_softmax_w, rows_delete, 1)

        logging.info('reading elmo vocabulary')
        if lemmatize_predictions:
            if os.path.isfile(vocab_path + '.lemmatized'):
                with open(vocab_path + '.lemmatized') as fin:
                    add_words_from_lines(fin)
            else:
                with open(vocab_path) as fin:
                    unlem = [x.strip() for x in fin.readlines()]
                logging.info('lemmatizing ELMo vocabulary')
                print('lemmatizing ELMo vocabulary')
                import spacy
                nlp = spacy.load("en", disable=['ner', 'parser'])
                new_vocab = []
                for spacyed in tqdm(
                        nlp.pipe(unlem, batch_size=1000, n_threads=multiprocessing.cpu_count()),
                        total=len(unlem)):
                    new_vocab.append(spacyed[0].lemma_ if spacyed[0].lemma_ != '-PRON-' else spacyed[0].lower_)
                with open(vocab_path + '.lemmatized', 'w') as fout:
                    for word in new_vocab:
                        fout.write('%s\n' % word)
                add_words_from_lines(new_vocab)
                logging.info('lemmatization done and cached to file')
                print('lemmatization done and cached to file')
        else:
            # no lemmatization
            with open(vocab_path) as fin:
                add_words_from_lines(fin)

        logging.info('caching cnn embeddings')
        # self.elmo.elmo_bilm.create_cached_cnn_embeddings(self.elmo_word_vocab)
        # self.elmo.elmo_bilm._has_cached_vocab = True

        self.elmo_word_vocab = self.elmo_word_vocab[:cutoff_elmo_vocab]
        self.elmo_softmax_w = self.elmo_softmax_w[:, :cutoff_elmo_vocab]

    def _warm_up_elmo(self):
        warm_up_sent = "Well , the line comes from deciding what the First Amendment interest is , " \
                       "and if this Court heed the First Amendment interest off of this difference " \
                       "between selecting who gets the benefit of 20 years of extension and just " \
                       "simply legislating in a general way prospectively , then this Court could " \
                       "hold , with respect to the prospective , that it 's not even necessary to " \
                       "raise the intermediate scrutiny in that context , but again , for Ashwander " \
                       "reasons we do n't think that this Court should address the prospective aspect " \
                       "of the CTEA even under the First Amendment .".split()
        for _ in range(3):
            _ = list(self.elmo.embed_sentences([warm_up_sent] * self.batch_size, self.batch_size))

    def _get_top_words_dist(self, state):
        log_probs = np.matmul(state, self.elmo_softmax_w)
        top_k_log_probs = np.argpartition(-log_probs, self.cutoff)[: self.cutoff]
        top_k_log_probs_vals = log_probs[top_k_log_probs]
        e_x = np.exp(top_k_log_probs_vals - np.max(top_k_log_probs_vals))
        probs = e_x / e_x.sum(axis=0)
        return top_k_log_probs, probs

    def predict_sent_substitute_representatives(self, inst_id_to_sentence: Dict[str, Tuple[List[str], int]],
                                                n_representatives: int,
                                                samples_per_side_per_representative: int) -> Dict[
        str, List[Dict[str, int]]]:
        """
        a representative is a dictionary made out of samples from both sides of the BiLM, predicting substitutes
        for a contextualized token.
        an example might look like:
        {'forward_jump':2,'backward_leap':1, 'backward_climb':1} (samples_per_side_per_representative=2)
        we return a list of n_representatives of those

        :param tokens: list of one sentence tokens
        :param target_idx: index of disambiguated token
        :param n_representatives: number of representatives
        :param samples_per_side_per_representative: number of samples to draw from each side
        :return:
        """
        inst_id_sent_tuples = list(inst_id_to_sentence.items())
        target = inst_id_sent_tuples[0][0].rsplit('.', 1)[0]
        lemma = inst_id_sent_tuples[0][0].split('.')[0]
        to_embed = []
        if self.disable_symmetric_patterns:
            # w/o sym. patterns - predict for blanked out word.
            # if the target word is the first or last in sentence get empty prediction by embedding '.'
            for _, (tokens, target_idx) in inst_id_sent_tuples:
                forward = tokens[:target_idx]
                backward = tokens[target_idx + 1:]
                if not forward:
                    forward = ['.']
                if not backward:
                    backward = ['.']
                to_embed += [forward, backward]
        else:
            # w/ sym. patterns - include target word + "and" afterwards in both directions
            for _, (tokens, target_idx) in inst_id_sent_tuples:
                to_embed += [tokens[:target_idx + 1] + ['and'], ['and'] + tokens[target_idx:]]
        logging.info('embedding %d sentences for target %s' % (len(to_embed), target))
        embedded = list(self.elmo.embed_sentences(to_embed, self.batch_size))

        results = {}
        for i in range(len(inst_id_sent_tuples)):
            inst_id, (tokens, target_idx) = inst_id_sent_tuples[i]
            sentence = ' '.join([t if i != target_idx else '***%s***' % t for i, t in enumerate(tokens)])
            logging.info('instance %s sentence: %s' % (inst_id, sentence))

            forward_out_em = embedded[i * 2][2, -1, :512]
            backward_out_em = embedded[i * 2 + 1][2, 0, 512:]

            forward_idxs, forward_dist = self._get_top_words_dist(forward_out_em)
            backward_idxs, backward_dist = self._get_top_words_dist(backward_out_em)

            forward_samples = []
            # after removing samples equal to disamb. target,
            # we might end up with not enough samples, so repeat until we have enough samples
            while len(forward_samples) < n_representatives * samples_per_side_per_representative:
                new_samples = list(
                    np.random.choice(forward_idxs, n_representatives * samples_per_side_per_representative * 2,
                                     p=forward_dist))
                new_samples = [x for x in new_samples if self.elmo_word_vocab[x].lower() != lemma]
                forward_samples += new_samples

            backward_samples = []
            while len(backward_samples) < n_representatives * samples_per_side_per_representative:
                new_samples = list(
                    np.random.choice(backward_idxs, n_representatives * samples_per_side_per_representative * 2,
                                     p=backward_dist))
                new_samples = [x for x in new_samples if self.elmo_word_vocab[x].lower() != lemma]
                backward_samples += new_samples
            representatives = []
            for _ in range(n_representatives):
                representative = {}
                for _ in range(samples_per_side_per_representative):
                    forward_sampled_word = self.elmo_word_vocab[forward_samples.pop()]
                    backward_sampled_word = self.elmo_word_vocab[backward_samples.pop()]
                    representative['fw:%s' % forward_sampled_word] = representative.get(
                        'fw:%s' % forward_sampled_word, 0) + 1
                    representative['bw:%s' % backward_sampled_word] = representative.get(
                        'bw:%s' % backward_sampled_word, 0) + 1
                representatives.append(representative)
            logging.info('first 3 representatives out of %d:\n%s' % (n_representatives, representatives[:3]))
            results[inst_id] = representatives
        return results
