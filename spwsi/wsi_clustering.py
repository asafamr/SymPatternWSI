from spwsi.bilm_interface import Bilm
from typing import Dict, List, Tuple
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.pipeline import make_pipeline
from collections import Counter
import logging


class WsiClusterer:
    def __init__(self, lm: Bilm, n_representatives_per_sentence: int, n_samples_per_side: int,
                 disable_tfidf: bool = False,number_of_clusters:int=6):
        self.lm = lm
        self.n_representatives_per_sentence = n_representatives_per_sentence
        self.n_samples_per_side = n_samples_per_side
        self.disable_tfidf=disable_tfidf
        self.number_of_clusters=number_of_clusters

    def soft_cluster_sentences_to_senses(self, inst_id_to_sent: Dict[str, Tuple[List[str], int]]):
        inst_ids_to_representatives = self.lm.predict_sent_substitute_representatives(inst_id_to_sent,
                                                                                      self.n_representatives_per_sentence,
                                                                                      self.n_samples_per_side)

        inst_ids_ordered = list(inst_ids_to_representatives.keys())
        lemma = inst_ids_ordered[0].rsplit('.', 1)[0]
        logging.info('clustering lemma %s' % lemma)
        representatives = [y for x in inst_ids_ordered for y in inst_ids_to_representatives[x]]
        in_pipline = [DictVectorizer()]
        if not self.disable_tfidf:
            in_pipline.append(TfidfTransformer())
        vectorizer = make_pipeline(*in_pipline)
        transformed = vectorizer.fit_transform(representatives).todense()
        clustering = AgglomerativeClustering(n_clusters=self.number_of_clusters, linkage='average', affinity='cosine')
        clustering.fit(transformed)
        senses = {}
        for i, inst_id in enumerate(inst_ids_ordered):
            inst_id_clusters = Counter(clustering.labels_[i * self.n_representatives_per_sentence:(
                                                                                                          i + 1) * self.n_representatives_per_sentence])
            senses[inst_id] = dict(inst_id_clusters)
        return senses
