from typing import Dict, List
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.pipeline import make_pipeline
from collections import Counter
import logging


def cluster_inst_ids_representatives(inst_ids_to_representatives: Dict[str, List[Dict[str, int]]],
                                     n_clusters: int, disable_tfidf: bool) -> Dict[str, Dict[str, int]]:
    """
    preforms agglomerative clustering on representatives of one SemEval target
    :param inst_ids_to_representatives: map from SemEval instance id to list of representatives
    :param n_clusters: fixed number of clusters to use
    :param disable_tfidf: disable tfidf processing of feature words
    :return: map from SemEval instance id to soft membership of clusters and their weight
    """
    inst_ids_ordered = list(inst_ids_to_representatives.keys())
    lemma = inst_ids_ordered[0].rsplit('.', 1)[0]
    logging.info('clustering lemma %s' % lemma)
    representatives = [y for x in inst_ids_ordered for y in inst_ids_to_representatives[x]]
    n_represent = len(representatives) // len(inst_ids_ordered)
    to_pipeline = [DictVectorizer()]
    if not disable_tfidf:
        to_pipeline.append(TfidfTransformer())
    data_transformer = make_pipeline(*to_pipeline)
    transformed = data_transformer.fit_transform(representatives).todense()
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='average', affinity='cosine')
    clustering.fit(transformed)
    senses = {}
    for i, inst_id in enumerate(inst_ids_ordered):
        inst_id_clusters = Counter(clustering.labels_[i * n_represent:
                                                      (i + 1) * n_represent])
        senses[inst_id] = dict([('%s.sense.%d' % (lemma, k), v) for (k, v) in inst_id_clusters.most_common()])
    return senses
