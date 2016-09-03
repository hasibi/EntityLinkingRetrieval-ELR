"""
Class for query annotations in the json file.

@author: Faegheh Hasibi (faegheh.hasibi@idi.ntnu.no)
"""
from nordlys.retrieval.lucene_tools import Lucene


class QueryAnnot(object):
    def __init__(self, annotations, score_th, qid=None):
        self.annotations = annotations
        self.score_th = score_th
        self.qid = qid
        self.__E = None
        self.__T = None
        self.__mentions = None

    @property
    def query(self):
        return self.annotations.get('query', None)

    @property
    def field_mappings(self):
        """Returns field mappings."""
        return self.annotations.get('field_mappings', {})

    @field_mappings.setter
    def field_mappings(self, value):
        if "field_mappings" not in self.annotations:
            self.annotations['field_mappings'] = {}
        self.annotations['field_mappings'].update(value)

    @property
    def E(self):
        """Returns set of annotated entities."""
        if self.__E is None:
            self.__E = {}
            for interpretation in self.annotations['interpretations'].values():
                for annot in interpretation['annots'].values():
                    if float(annot['score']) >= self.score_th:
                        self.__E[annot['uri']] = annot['score']
        return self.__E

    @property
    def T(self):
        """Returns all query terms."""
        if self.__T is None:
            analyzed_query = Lucene.preprocess(self.query)
            self.__T = analyzed_query.split(" ")
        return self.__T

    @property
    def mentions(self):
        """Returns all mentions (among all annotations)."""
        if self.__mentions is None:
            self.__mentions = {}
            for interpretation in self.annotations['interpretations'].values():
                for mention, annot in interpretation['annots'].iteritems():
                    if float(annot['score']) >= self.score_th:
                        analyzed_phrase = Lucene.preprocess(mention)
                        if (analyzed_phrase is not None) and (analyzed_phrase.strip() != ""):
                            self.__mentions[analyzed_phrase] = annot['score']
        return self.__mentions

    def get_all_phrases(self):
        """Returns phrases for the ordered part of the model. (bigram and n-gram of mentions)"""
        all_phrases = set()
        for s_t in self.mentions:
            if len(s_t.split(" ")) > 1:
                all_phrases.add(s_t)
        analyzed_query = Lucene.preprocess(self.query)
        query_terms = analyzed_query.split(" ")
        for i in range(0, len(query_terms)-1):
            bigram = " ".join([query_terms[i], query_terms[i+1]])
            all_phrases.add(bigram)
        return all_phrases

    def update(self, key, value):
        """Updates the annotation."""
        self.annotations[key] = value
