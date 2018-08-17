from typing import List, Tuple, Dict


class Bilm:
    """
    implement this interface for to use a custom biLM
    """

    def __init__(self):
        pass

    def predict_sent_substitute_representatives(self, inst_id_to_sentence: Dict[str, Tuple[List[str], int]],
                                                n_represent: int, n_samples_side: int, disable_symmetric_patterns: bool,
                                                disable_lemmatiziation: bool, prediction_cutoff: int) \
            -> Dict[str, List[Dict[str, int]]]:
        raise NotImplementedError()
