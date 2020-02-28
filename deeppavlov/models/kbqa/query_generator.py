import json
import re
import nltk
from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable
from typing import Union
from deeppavlov.models.kbqa.template_matcher import TemplateMatcher
from deeppavlov.models.kbqa.entity_linking_cq import EntityLinkerCQ
from deeppavlov.models.kbqa.wiki_parser import WikiParser
from deeppavlov.models.kbqa.rel_ranking_infer import RelRankerInfer
fl = open("/home/evseev/LC-QUAD2.0/test.json").read()
dataset = json.loads(fl)

@register('query_generator')
class QueryGenerator(Component, Serializable):
    def __init__(self, template_matcher: TemplateMatcher,
                       #linker: EntityLinkerCQ,
                       wiki_parser: WikiParser,
                       rel_ranker: RelRankerInfer,
                       load_path: str,
                       rank_rels_filename_1: str,
                       rank_rels_filename_2: str, **kwargs):
     
        super().__init__(save_path=None, load_path=load_path)
        self.template_matcher = template_matcher
        #self.linker = linker
        self.wiki_parser = wiki_parser
        self.rel_ranker = rel_ranker
        self.rank_rels_filename_1 = rank_rels_filename_1
        self.rank_rels_filename_2 = rank_rels_filename_2
        self.load()

    def load(self) -> None:
        with open(self.load_path / self.rank_rels_filename_1, 'r') as fl1:
            lines = fl1.readlines()
            self.rank_list_0 = [line.split('\t')[0] for line in lines]

        with open(self.load_path / self.rank_rels_filename_2, 'r') as fl2:
            lines = fl2.readlines()
            self.rank_list_1 = [line.split('\t')[0] for line in lines]

    def save(self) -> None:
        pass

    def __call__(self, question_tuple, template_type, entities_from_ner):
        question = question_tuple[0]
        self.template_num  = template_type[0]
        
        #entity_ids = [self.linker(entity)[:10] for entity in entities]
        '''
        print("question", question)
        print("template_type", template_type)
        print("entities", entities)
        print("entity_ids", entity_ids)
        '''
        question = question.replace('"', "'").replace('{', '').replace('}', '').replace('  ', ' ')
        entities_from_template, rels_from_template = self.template_matcher(question)
        entities = entities_from_template if entities_from_template else entities_from_ner
        #entity_ids = [self.linker(entity)[:10] for entity in entities]
       
        self.template_num = 6
        entity_ids = [["Q11173"], ["Q29006389"]]

        if self.template_num == 0 or self.template_num == 1:
            candidate_outputs = self.complex_question_with_number_solver(question, entity_ids)

        if self.template_num == 2 or self.template_num == 3:
            candidate_outputs = self.complex_question_with_qualifier_solver(question, entity_ids)

        if self.template_num == 4:
            candidate_outputs = self.questions_with_count_solver(question, entity_ids)

        if self.template_num == 5:
            candidate_outputs = self.maxmin_one_entity_solver(question, entity_ids[0][:5])

        if self.template_num == 6:
            candidate_outputs = self.maxmin_two_entities_solver(question, entity_ids)

        if self.template_num == 7:
            candidate_outputs = self.two_hop_solver(question, entity_ids, rels_from_template)

        print(candidate_outputs)

        return candidate_outputs


    def complex_question_with_number_solver(self, question, entity_ids):
        question_tokens = nltk.word_tokenize(question)
        ex_rels = []
        for entity in entity_ids[0]:
            ex_rels += self.wiki_parser("rels", "forw", entity, type_of_rel="direct")
        ex_rels = list(set(ex_rels))
        scores = self.rel_ranker(question, ex_rels)
        top_rels = [score[0] for score in scores]
        print(top_rels)
        year = self.extract_year(question_tokens, question)
        number = False
        if not year:
            number = self.extract_number(question_tokens, question)
        print(year, number)

        candidate_outputs = []
            
        if year:
            candidate_outputs = self.find_relevant_subgraph_cqwn(entity_ids[0][:5], top_rels[:7], year)
        if number:
            candidate_outputs = self.find_relevant_subgraph_cqwn(entity_ids[0][:5], top_rels[:7], number)

        return candidate_outputs

    def complex_question_with_qualifier_solver(self, question, entity_ids):
        ex_rels = []
        for entity in entity_ids[0]:
            ex_rels += self.wiki_parser("rels", "forw", entity, type_of_rel="direct")
        ex_rels = list(set(ex_rels))
        scores = self.rel_ranker(question, ex_rels)
        top_rels = [score[0] for score in scores]
        print("top_rels", top_rels)

        candidate_outputs = []

        if len(entity_ids) > 1:
            ent_combs = []
            for n, entity_1 in enumerate(entity_ids[0]):
                for m, entity_2 in enumerate(entity_ids[1]):
                    ent_combs.append((entity_1, entity_2, (n+m)))
                    ent_combs.append((entity_2, entity_1, (n+m)))

            ent_combs = sorted(ent_combs, key=lambda x: x[2])

            candidate_outputs = self.find_relevant_subgraph_cqwq(ent_combs, top_rels[:5])

        return candidate_outputs

    def questions_with_count_solver(self, question, entity_ids):
        candidate_outputs = []

        ex_rels = []
        for entity_id in entity_ids:
            for entity in entity_id[:5]:
                ex_rels += self.wiki_parser("rels", "forw", entity, type_of_rel="direct")
                ex_rels += self.wiki_parser("rels", "backw", entity, type_of_rel="direct")

        ex_rels = list(set(ex_rels))
        scores = self.rel_ranker(question, ex_rels)
        top_rels = [score[0] for score in scores]
        print("top_rels", top_rels)
        answers = []
        for entity_id in entity_ids:
            for entity in entity_id[:5]:
                for rel in top_rels:
                    answers += self.wiki_parser("objects", "forw", entity, top_rels[0], type_of_rel="direct")
                    if len(answers) > 0:
                        candidate_outputs.append((rel, len(answers)))
                    else:
                        answers += self.wiki_parser("objects", "backw", entity, top_rels[0], type_of_rel="direct")
                        candidate_outputs.append((rel, len(answers)))

        return candidate_outputs
    
    def maxmin_one_entity_solver(self, question, entities_list):
        scores = self.rel_ranker(question, self.rank_list_0)
        top_rels = [score[0] for score in scores]
        print("top_rels", top_rels)
        ascending = self.asc_desc(question)
        candidate_outputs = self.find_relevant_subgraph_maxmin_one(entities_list, top_rels[:5])
        reverse = False
        if ascending:
            reverse = True
        candidate_outputs = sorted(candidate_outputs, key=lambda x: x[1], reverse=reverse)

        return candidate_outputs

    def maxmin_two_entities_solver(self, question, entity_ids):
        ex_rels = []
        for entities_list in entity_ids:
            for entity in entities_list:
                ex_rels += self.wiki_parser("rels", "backw", entity, type_of_rel="direct")
        
        ex_rels = list(set(ex_rels))
        scores_1 = self.rel_ranker(question, ex_rels)
        top_rels_1 = [score[0] for score in scores_1]
        #top_rels_1 = ["P527"]
        print("top_rels_1", top_rels_1)

        scores_2 = self.rel_ranker(question, self.rank_list_1)
        top_rels_2 = [score[0] for score in scores_2]
        top_rels_2 = ["P2658"]
        print("top_rels_2", top_rels_2)

        candidate_outputs = []

        if len(entity_ids) > 1:
            ent_combs = []
            for n, entity_1 in enumerate(entity_ids[0]):
                for m, entity_2 in enumerate(entity_ids[1]):
                    ent_combs.append((entity_1, entity_2, (n+m)))
                    ent_combs.append((entity_2, entity_1, (n+m)))

            ent_combs = sorted(ent_combs, key=lambda x: x[2])

            candidate_outputs = self.find_relevant_subgraph_maxmin_two(ent_combs, top_rels_1[:5], top_rels_2[:5])

            ascending = self.asc_desc(question)
            reverse = False
            if ascending:
                reverse = True
            candidate_outputs = sorted(candidate_outputs, key=lambda x: x[1], reverse=reverse)

        return candidate_outputs

    def two_hop_solver(self, question, entity_ids, rels_from_template=None):
        candidate_outputs = []
        if len(entity_ids) == 1:
            if rels_from_template is not None:
                if len(rels_from_template) == 1:
                    relation = rels_from_template[0][0]
                    direction = rels_from_template[0][1]
                    objects = self.wiki_parser("objects", direction, entity[0][0], relation, type_of_rel="direct")
                    if objects:
                        candidate_outputs.append((relation, objects[0]))
                    
            else:
                ex_rels = []
                for entity in entity_ids[0][:5]:
                    ex_rels += self.wiki_parser("rels", "forw", entity, type_of_rel="direct")
                    ex_rels += self.wiki_parser("rels", "backw", entity, type_of_rel="direct")

                ex_rels = list(set(ex_rels))
                scores = self.rel_ranker(question, ex_rels)
                top_rels = [score[0] for score in scores]

                ex_rels_2 = []
                for entity in entity_ids[0][:5]:
                    for rel in top_rels:
                        objects_mid = self.wiki_parser("objects", "forw", entity, rel, type_of_rel="direct")
                        objects_mid += self.wiki_parser("objects", "backw", entity, rel, type_of_rel="direct")
                        if len(objects_mid) < 10:
                            for obj in objects_mid:
                                ex_rels_2 += self.wiki_parser("rels", "forw", obj, type_of_rel="direct")

                ex_rels_2 = list(set(ex_rels_2))
                scores_2 = self.rel_ranker(question, ex_rels_2)
                top_rels_2 = [score[0] for score in scores_2]

                for rel in top_rels:
                    candidate_outputs.append([rel])

                for rel_1 in top_rels:
                    for rel_2 in top_rels_2:
                        candidate_outputs.append([rel_1, rel_2])
            
        return candidate_outputs

    def find_relevant_subgraph_cqwn(self, entities_list, rels, num):
        candidate_outputs = []

        for entity in entities_list:
            for rel in rels:
                objects_1 = self.wiki_parser("objects", "forw", entity, rel, type_of_rel=None)
                print("objects_1", objects_1)
                for obj in objects_1:
                    if self.template_num == 0:
                        answers = self.wiki_parser("objects", "forw", obj, rel, type_of_rel="statement")
                        print("answers", answers)
                        second_rels = self.wiki_parser("rels", "forw", obj, type_of_rel="qualifier", filter_obj=num)
                        print("second_rels", second_rels)
                        if len(second_rels) > 0 and len(answers) > 0:
                            for second_rel in second_rels:
                                for ans in answers:
                                    candidate_outputs.append((rel, second_rel, ans))
                    if self.template_num == 1:
                        answer_triplets = self.wiki_parser("triplets", "forw", obj, type_of_rel="qualifier")
                        print("answer_triplets", answer_triplets)
                        second_rels = self.wiki_parser("rels", "forw", obj, rel,
                                                       type_of_rel="statement", filter_obj=num)
                        print("second_rels", second_rels)
                        if len(second_rels) > 0 and len(answer_triplets) > 0:
                            for ans in answer_triplets:
                                candidate_outputs.append((rel, ans[1], ans[2]))
                
        return candidate_outputs

    def find_relevant_subgraph_cqwq(self, ent_combs, rels):
        candidate_outputs = []
        
        for ent_comb in ent_combs:
            for rel in rels:
                objects_1 = self.wiki_parser("objects", "forw", ent_comb[0], rel, type_of_rel=None)
                print("objects_1", objects_1)
                for obj in objects_1:
                    if self.template_num == 2:
                        answer_triplets = self.wiki_parser("triplets", "forw", obj, type_of_rel="qualifier")
                        print("answer_triplets", answer_triplets)
                        second_rels = self.wiki_parser("rels", "backw", ent_comb[1], rel, obj, type_of_rel="statement")
                        print("second_rels", second_rels)
                        if len(second_rels) > 0 and len(answer_triplets) > 0:
                            for ans in answer_triplets:
                                candidate_outputs.append((rel, ans[1], ans[2]))
                    if self.template_num == 3:
                        answers = self.wiki_parser("objects", "forw", obj, rel, type_of_rel="statement")
                        second_rels = self.wiki_parser("rels", "backw", ent_comb[1], rel=None,
                                                       obj=obj, type_of_rel="qualifier")
                        if len(second_rels) > 0 and len(answers) > 0:
                            for second_rel in second_rels:
                                for ans in answers:
                                    candidate_outputs.append((rel, second_rel, ans))
                
        return candidate_outputs

    def find_relevant_subgraph_maxmin_one(self, entities_list, rels):
        candidate_answers = []

        for entity in entities_list:
            objects_1 = self.wiki_parser("objects", "backw", entity, "P31", type_of_rel="direct")
            for rel in rels:
                candidate_answers = []
                for obj in objects_1:
                    objects_2 = self.wiki_parser("objects", "forw", obj, rel, type_of_rel="direct", filter_obj="http://www.w3.org/2001/XMLSchema#decimal")
                    if len(objects_2) > 0:
                        number = re.search(r'["]([^"]*)["]*', objects_2[0]).group(1)
                        candidate_answers.append((obj, float(number)))
                
                if len(candidate_answers) > 0:
                    return candidate_answers

        return candidate_answers

    def find_relevant_subgraph_maxmin_two(self, ent_combs, rels_1, rels_2):
        candidate_answers = []

        for ent_comb in ent_combs:
            objects_1 = self.wiki_parser("objects", "backw", ent_comb[0], "P31", type_of_rel="direct")
            #print("objects_1", objects_1)
            for rel_1 in rels_1:
                objects_2 = self.wiki_parser("objects", "backw", ent_comb[1], rel_1, type_of_rel="direct")
                #print("objects_2", objects_2)
                objects_intersect = list(set(objects_1) & set(objects_2))
                print("objects_intersect", objects_intersect)
                for rel_2 in rels_2:
                    candidate_answers = []
                    for obj in objects_intersect:
                        objects_3 = self.wiki_parser("objects", "forw", obj, rel_2, type_of_rel="direct", filter_obj="http://www.w3.org/2001/XMLSchema#decimal")
                        print("objects_3", objects_3)
                        if len(objects_3) > 0:
                            number = re.search(r'["]([^"]*)["]*', objects_3[0]).group(1)
                            candidate_answers.append((obj, float(number)))
                    
                    if len(candidate_answers) > 0:
                        return candidate_answers

        return candidate_answers

    def extract_year(self, question_tokens, question):
        year = ""
        fnd = re.search(r'.*\d/\d/(\d{4}).*', question)
        if fnd is not None:
            year = fnd.group(1)
        if len(year) == 0:
            fnd = re.search(r'.*\d\-\d\-(\d{4}).*', question)
            if fnd is not None:
                year = fnd.group(1)
        if len(year) == 0:
            fnd = re.search(r'.*(\d{4})\-\d\-\d.*', question)
            if fnd is not None:
                year = fnd.group(1)
        if len(year) == 0:
            for tok in question_tokens:
                isdigit = [l.isdigit() for l in tok[:4]]
                isdigit_0 = [l.isdigit() for l in tok[-4:]]
                
                if sum(isdigit) == 4 and len(tok) == 4:
                    year = tok
                    break
                if sum(isdigit) == 4 and len(tok) > 4 and tok[4] == '-':
                    year = tok[:4]
                    break
                if sum(isdigit_0) == 4 and len(tok) > 4 and tok[-5] == '-':
                    year = tok[-4:]
                    break

        return year

    def extract_number(self, question_tokens, question):
        number = ""
        fnd = re.search(r'.*(\d\.\d+e\+\d+)\D*', question)
        if fnd is not None:
            number = fnd.group(1)
        if len(number) == 0:
            for tok in question_tokens:
                if tok[0].isdigit():
                    number = tok
                    break

        number = number.replace('1st', '1').replace('2nd', '2').replace('3rd', '3')
        number = number.strip(".0")

        return number

    def asc_desc(self, question):
        question_lower = question.lower()
        max_words = ["maximum", "highest", "max(", "greatest", "most", "longest"]
        min_words = ["lowest", "smallest", "least", "min", "min("]
        for word in max_words:
            if word in question_lower:
                return False

        for word in min_words:
            if word in question_lower:
                return True

        return True

