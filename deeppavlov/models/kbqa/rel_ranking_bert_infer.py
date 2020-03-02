import pickle
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.models.ranking.rel_ranker import RelRanker


@register('rel_ranking_bert_infer')
class RelRankerBertInfer(Component, Serializable):
    def __init__(self, load_path: str, rel_q2name_filename: str,
                       ranker, batch_size: int = 32,  **kwargs):
        super().__init__(save_path=None, load_path=load_path)
        self.rel_q2name_filename = rel_q2name_filename
        self.ranker = ranker
        self.batch_size = batch_size
        self.load()

    def load(self) -> None:
        with open(self.load_path / self.rel_q2name_filename, 'rb') as inv:
            self.rel_q2name = pickle.load(inv)
            
    def save(self) -> None:
        pass

    def __call__(self, questions, candidate_answers):
        question = questions[0]
        answers_with_scores = []
        
        for i in range(len(candidate_answers)//self.batch_size):
            questions_batch = []
            rels_labels_batch = []
            answers_batch = []
            for j in range(self.batch_size):
                candidate_rels = candidate_answers[(i*self.batch_size+j)][:-1]
                candidate_answer = candidate_answers[(i*self.batch_size+j)][-1]
                candidate_rels = " [SEP] ".join([self.rel_q2name[candidate_rel] \
                                 for candidate_rel in candidate_rels if candidate_rel in self.rel_q2name])

                if candidate_rels:
                    questions_batch.append(question)
                    rels_labels_batch.append(candidate_rels)
                    answers_batch.append(candidate_answer)

            probas = self.ranker(questions_batch, rels_labels_batch)
            probas = [proba[1] for proba in probas]
            for j, answer in enumerate(answers_batch):
                answers_with_scores.append((answer, probas[j]))

        questions_batch = []
        rels_labels_batch = []
        answers_batch = []
        for j in range(len(candidate_answers)%self.batch_size):
            candidate_rels = candidate_answers[(len(candidate_rels)//self.batch_size*self.batch_size+j)][:-1]
            candidate_answer = candidate_answers[(len(candidate_rels)//self.batch_size*self.batch_size+j)][-1]
            candidate_rels = " [SEP] ".join([self.rel_q2name[candidate_rel] \
                                 for candidate_rel in candidate_rels if candidate_rel in self.rel_q2name])

            if candidate_rels:
                questions_batch.append(question)
                rels_labels_batch.append(candidate_rels)
                answers_batch.append(candidate_answer)

        print("len of rels_labels_batch", len(rels_labels_batch))
        probas = self.ranker(questions_batch, rels_labels_batch)
        probas = [proba[1] for proba in probas]
        for j, answer in enumerate(answers_batch):
            answers_with_scores.append((answer, probas[j]))

        answers_with_scores = sorted(answers_with_scores, key=lambda x: x[1], reverse=True)
        print(len(rels_with_scores))
        return answers_with_scores[0]
    

