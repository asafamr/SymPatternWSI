import spacy
import os
from xml.etree import ElementTree
from typing import Dict
import tempfile
import subprocess
import logging


def generate_sem_eval_2013(dir_path: str):
    logging.info('reading SemEval dataset from %s' % dir_path)
    nlp = spacy.load("en", disable=['ner', 'parser'])
    in_xml_path = os.path.join(dir_path, 'contexts/senseval2-format/semeval-2013-task-13-test-data.senseval2.xml')
    gold_key_path = os.path.join(dir_path, 'keys/gold/all.key')
    with open(in_xml_path, encoding="utf-8") as fin_xml, open(gold_key_path, encoding="utf-8") as fin_key:
        instid_in_key = set()
        for line in fin_key:
            lemma_pos, inst_id, _ = line.strip().split(maxsplit=2)
            instid_in_key.add(inst_id)
        et_xml = ElementTree.parse(fin_xml)
        for word in et_xml.getroot():
            for inst in word.getchildren():
                inst_id = inst.attrib['id']
                if inst_id not in instid_in_key:
                    # discard unlabeled instances
                    continue
                context = inst.find("context")
                before, target, after = list(context.itertext())
                before = [x.text for x in nlp(before.strip(), disable=['parser', 'tagger', 'ner'])]
                target = target.strip()
                after = [x.text for x in nlp(after.strip(), disable=['parser', 'tagger', 'ner'])]
                yield before + [target] + after, len(before), inst_id


def evaluate_labeling(dir_path, labeling: Dict[str, Dict[str, int]], key_path: str = None) \
        -> Dict[str, Dict[str, float]]:
    """
    labeling example : {'become.v.3': {'become.sense.1':3,'become.sense.5':17} ... }
    means instance become.v.3' is 17/20 in sense 'become.sense.5' and 3/20 in sense 'become.sense.1'
    :param key_path: write produced key to this file
    :param dir_path: SemEval dir
    :param labeling: instance id labeling
    :return: FNMI, FBC as calculated by SemEval provided code
    """
    logging.info('starting evaluation key_path: %s' % key_path)

    def get_scores(gold_key, eval_key):
        ret = {}
        for metric, jar, column in [
            #         ('jaccard-index','SemEval-2013-Task-13-test-data/scoring/jaccard-index.jar'),
            #         ('pos-tau', 'SemEval-2013-Task-13-test-data/scoring/positional-tau.jar'),
            #         ('WNDC', 'SemEval-2013-Task-13-test-data/scoring/weighted-ndcg.jar'),
            ('FNMI', os.path.join(dir_path, 'scoring/fuzzy-nmi.jar'), 1),
            ('FBC', os.path.join(dir_path, 'scoring/fuzzy-bcubed.jar'), 3),
        ]:
            logging.info('calculating metric %s' % metric)
            res = subprocess.Popen(['java', '-jar', jar, gold_key, eval_key], stdout=subprocess.PIPE).stdout.readlines()
            # columns = []
            for line in res:
                line = line.decode().strip()
                if line.startswith('term'):
                    # columns = line.split('\t')
                    pass
                else:
                    split = line.split('\t')
                    if len(split) > column:
                        word = split[0]
                        # results = list(zip(columns[1:], map(float, split[1:])))
                        result = split[column]
                        if word not in ret:
                            ret[word] = {}
                        ret[word][metric] = float(result)

        return ret

    with tempfile.NamedTemporaryFile('wt') as fout:
        lines = []
        for instance_id, clusters_dict in labeling.items():
            clusters = sorted(clusters_dict.items(), key=lambda x: x[1])
            clusters_str = ' '.join([('%s/%d' % (cluster_name, count)) for cluster_name, count in clusters])
            lemma_pos = instance_id.rsplit('.', 1)[0]
            lines.append('%s %s %s' % (lemma_pos, instance_id, clusters_str))
        fout.write('\n'.join(lines))
        fout.flush()
        scores = get_scores(os.path.join(dir_path, 'keys/gold/all.key'),
                            fout.name)
        if key_path:
            logging.info('writing key to file %s' % key_path)
            with open(key_path, 'w', encoding="utf-8") as fout2:
                fout2.write('\n'.join(lines))
        return scores
