import pickle
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.models.ranking.rel_ranker import RelRanker


@register('rel_ranking_infer')
class RelRankerInfer(Component, Serializable):
    def __init__(self, load_path: str, rel_q2name_filename: str,
                       ranker: RelRanker, rels_to_leave: int = 15, batch_size: int = 100, **kwargs):
        super().__init__(save_path=None, load_path=load_path)
        self.rel_q2name_filename = rel_q2name_filename
        self.ranker = ranker
        self.rels_to_leave = rels_to_leave
        self.batch_size = batch_size
        self.load()

    def load(self) -> None:
        with open(self.load_path / self.rel_q2name_filename, 'rb') as inv:
            self.rel_q2name = pickle.load(inv)
            
    def save(self) -> None:
        pass

    def __call__(self, question, candidate_rels):
        rels_with_scores = []
        print("candidate_rels", len(candidate_rels))
        print("num_cycles", len(candidate_rels)//self.batch_size)
        print("batch_size", self.batch_size)
        for i in range(len(candidate_rels)//self.batch_size):
            questions_batch = []
            rels_labels_batch = []
            rels_batch = []
            for j in range(self.batch_size):
                candidate_rel = candidate_rels[(i*self.batch_size+j)]
                if candidate_rel in self.rel_q2name:
                    questions_batch.append(question)
                    rels_batch.append(candidate_rel)
                    rels_labels_batch.append(self.rel_q2name[candidate_rel])
            probas = self.ranker(questions_batch, rels_labels_batch)
            probas = [proba[1] for proba in probas]
            for j, rel in enumerate(rels_batch):
                rels_with_scores.append((rel, probas[j]))

        questions_batch = []
        rels_batch = []
        rels_labels_batch = []
        for j in range(len(candidate_rels)%self.batch_size):
            candidate_rel = candidate_rels[(len(candidate_rels)//self.batch_size*self.batch_size+j)]
            if candidate_rel in self.rel_q2name:
                questions_batch.append(question)
                rels_batch.append(candidate_rel)
                rels_labels_batch.append(self.rel_q2name[candidate_rel])
        probas = self.ranker(questions_batch, rels_labels_batch)
        probas = [proba[1] for proba in probas]
        for j, rel in enumerate(rels_batch):
            rels_with_scores.append((rel, probas[j]))

        rels_with_scores = sorted(rels_with_scores, key=lambda x: x[1], reverse=True)
        print(len(rels_with_scores))
        return rels_with_scores[:self.rels_to_leave]

