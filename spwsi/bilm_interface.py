from typing import List, Tuple, Dict


class Bilm:
    """
    implement this interface for to use a custom BiLM
    """
    def __init__(self):
        pass

    def predict_sent_substitute_representatives(self, inst_id_to_sentence: Dict[str, Tuple[List[str], int]],
                                                n_representatives: int,
                                                samples_per_side_per_representative: int) -> Dict[
        str, List[Dict[str, int]]]:
        raise NotImplementedError()
