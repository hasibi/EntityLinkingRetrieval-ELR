"""
ELR extension of MRF based models: LM, MLM, PRMS, SDM, and FSDM

@author: Faegheh Hasibi (faegheh.hasibi@idi.ntnu.no)
"""

from __future__ import division

import math

from nordlys.elr.field_mapping import FieldMapping
from nordlys.elr.top_fields import TopFields
from nordlys.retrieval.lucene_tools import Lucene
from nordlys.retrieval.scorer import ScorerLM


class ScorerMRF(object):
    DEBUG = 0

    TERM = "terms"
    ORDERED = "ordered"
    UNORDERED = "unordered"
    URI = "uris"
    SLOP = 6  # Window = 8

    def __init__(self, lucene_term, lucene_uri, params, query_annot):
        self.lucene_term = lucene_term
        self.lucene_uri = lucene_uri
        self.params = params
        self.query_annot = query_annot
        self.phrase_freq = {}

        self.scorer_lm_term = ScorerLM(self.lucene_term, None, {'smoothing_method': "dirichlet"})
        self.scorer_lm_uri = ScorerLM(self.lucene_uri, None, {})
        self.instance_list = []
        self.__n_fields = None
        self.__bigrams = None
        self.__mlm_all_mapping = None

    @property
    def n_fields(self):
        """Returns number of fields for fielded models."""
        if self.__n_fields is None:
            model = self.params['model']
            if ("prms" in model) or ("fsdm" in model) or ("mlm-all" in model):
                self.__n_fields = 10 if self.params['n_fields'] is None else self.params['n_fields']
        return self.__n_fields

    @property
    def bigrams(self):
        """Returns all query bigrams."""
        if self.__bigrams is None:
            self.__bigrams = []
            for i in range(0, len(self.query_annot.T)-1):
                bigram = " ".join([self.query_annot.T[i], self.query_annot.T[i+1]])
                self.__bigrams.append(bigram)
        return self.__bigrams

    @property
    def mlm_all_mapping(self):
        if self.__mlm_all_mapping is None:
            self.__mlm_all_mapping = {}
            fields = TopFields(self.lucene_term).get_top_index(self.n_fields)
            weight = 1.0 / len(fields)
            for field in fields:
                self.__mlm_all_mapping[field] = weight
        return self.__mlm_all_mapping

    @staticmethod
    def get_scorer(lucene_term, lucene_uri, params, query_annot):
        """
        Returns Scorer object (Scorer factory).

        :param lucene_term: Lucene object for terms
        :param lucene_uri: Lucene object for uris
        :param params: dict with models parameters
        :param query_annot: query annotation with the mapping probabilities
        """
        model = params['model']
        lambd = params['lambda']
        print "\t" + model + " scoring ..."
        if (model == "lm") or (model == "prms") or (model == "mlm-all") or (model == "mlm-tc"):
            params['lambda'] = [1.0, 0.0, 0.0] if lambd is None else lambd
            return ScorerFSDM(lucene_term, lucene_uri, params, query_annot)
        elif (model == "sdm") or (model == "fsdm"):
            params['lambda'] = [0.8, 0.1, 0.1] if lambd is None else lambd
            return ScorerFSDM(lucene_term, lucene_uri, params, query_annot)
        elif (model == "lm_elr") or (model == "prms_elr") or (model == "mlm-tc_elr") or (model == "mlm-all_elr"):
            params['lambda'] = [0.9, 0.0, 0.0, 0.1] if lambd is None else lambd
            return ScorerELR(lucene_term, lucene_uri, params, query_annot)
        elif (model == "sdm_elr") or (model == "fsdm_elr"):
            params['lambda'] = [0.8, 0.05, 0.05, 0.1] if lambd is None else lambd
            return ScorerELR(lucene_term, lucene_uri, params, query_annot)
        else:
            raise Exception("Unknown model '" + model + "'")

    def get_field_weights(self, clique_type, c):
        """
        Returns field mappings

        :param clique_type: [TERM | ORDERED | UNORDERED | URI]
        :param c: str (term, phrase, or uri)
        :return: {field: prob}
        """
        model = self.params['model']
        if (model == "lm") or (model == "lm_elr") or (model == "sdm") or (model == "sdm_elr"):
            return {Lucene.FIELDNAME_CONTENTS: 1}
        elif (model == "prms") or (model == "prms_elr") or (model == "fsdm") or (model == "fsdm_elr"):
            return self.get_prms_mapping(clique_type)[c]
        elif (model == "mlm-tc") or (model == "mlm-tc_elr"):
            if clique_type == self.URI:
                return self.get_prms_mapping(clique_type)[c]
            else:
                return {'names': 0.2, 'contents': 0.8}
        elif (model == "mlm-all") or (model == "mlm-all_elr"):
            if clique_type == self.URI:
                return self.get_prms_mapping(clique_type)[c]
            else:
                return self.mlm_all_mapping

    def get_prms_mapping(self, clique_type):
        """
        Gets PRMS mapping probability for a clique type

        :param clique_type: [TERM | ORDERED | UNORDERED | URI]
        :return Dictionary {phrase: {field: weight, ..}, ..}
        """
        if clique_type not in self.query_annot.field_mappings:
            mapper = FieldMapping(self.lucene_term, self.lucene_uri, self.n_fields)
            if clique_type == self.TERM:
                self.query_annot.field_mappings = {clique_type: mapper.get_mapping_terms(set(self.query_annot.T))}
            elif clique_type == self.ORDERED:
                self.query_annot.field_mappings = {clique_type: mapper.get_mapping_phrases(set(self.bigrams), 0, True)}
            elif clique_type == self.UNORDERED:
                self.query_annot.field_mappings = {clique_type: mapper.get_mapping_phrases(set(self.bigrams),
                                                                                           self.SLOP, False)}
            elif clique_type == self.URI:
                self.query_annot.field_mappings = {clique_type: mapper.get_mapping_uris(set(self.query_annot.E))}
        return self.query_annot.field_mappings[clique_type]

    def set_phrase_freq(self, clique_type, c, fields):
        """Sets document and collection frequency for phrase."""
        if clique_type not in self.phrase_freq:
            self.phrase_freq[clique_type] = {}
        if c not in self.phrase_freq.get(clique_type, {}):
            self.phrase_freq[clique_type][c] = {}
            for f in fields:
                if clique_type == self.ORDERED:
                    doc_freq = self.lucene_term.get_doc_phrase_freq(c, f, 0, True)
                elif clique_type == self.UNORDERED:
                    doc_freq = self.lucene_term.get_doc_phrase_freq(c, f, self.SLOP, False)

                self.phrase_freq[clique_type][c][f] = doc_freq
                self.phrase_freq[clique_type][c][f]['coll_freq'] = sum(doc_freq.values())

    @staticmethod
    def normalize_el_scores(scores):
        """Normalize entity linking score, so that sum of all scores equal to 1"""
        normalized_scores = {}
        sum_score = sum(scores.values())
        for item, score in scores.iteritems():
            normalized_scores[item] = score / sum_score
        return normalized_scores

    def get_p_t_d(self, t, field_weights, doc_id):
        """
        p(t|d) = sum_{f in F} p(t|d_f) p(f|t)

        :param t: term
        :param field_weights: Dictionary {f: p_f_t, ...}
        :param doc_id: entity id
        :return  p(t|d)
        """
        lucene_doc_id_t = self.lucene_term.get_lucene_document_id(doc_id)
        p_t_d = 0
        for f, p_f_t in field_weights.iteritems():
            if self.DEBUG:
                print "\tt:", t, "f:", f
            p_t_d_f = self.scorer_lm_term.get_term_prob(lucene_doc_id_t, f, t)
            p_t_d += p_t_d_f * p_f_t
            if self.DEBUG:
                print "\t\tp(t|d_f):", p_t_d_f, "p(f|t):", p_f_t, "p(t|d_f).p(f|t):", p_t_d_f * p_f_t
        if self.DEBUG:
            print "\tp(t|d):", p_t_d
        return p_t_d

    def get_p_o_d(self, o, field_weights, doc_id):
        """
        p(o|d) = sum_{f in F} p(o|d_f) p(f|o) for ordered search

        :param o: phrase (ordered search)
        :param field_weights: Dictionary {f: p_f_o, ...}
        :param doc_id: entity id
        :return  p(o|d)
        """
        lucene_doc_id_t = self.lucene_term.get_lucene_document_id(doc_id)
        self.set_phrase_freq(self.ORDERED, o, field_weights)
        p_o_d = 0
        for f, p_f_o in field_weights.iteritems():
            if self.DEBUG:
                print "\to:", o, "f:", f
            tf_t_d_f = self.phrase_freq[self.ORDERED][o].get(f, {}).get(doc_id, 0)
            tf_t_C_f = self.phrase_freq[self.ORDERED][o].get(f, {}).get('coll_freq', 0)
            p_o_d_f = self.scorer_lm_term.get_term_prob(lucene_doc_id_t, f, o, tf_t_d_f=tf_t_d_f, tf_t_C_f=tf_t_C_f)
            p_o_d += p_o_d_f * p_f_o
            if self.DEBUG:
                print "\t\tp(o|d_f):", p_o_d_f, "p(f|o):", p_f_o, "p(o|d_f).p(f|o):", p_o_d_f * p_f_o
        if self.DEBUG:
            print "\tp(o|d):", p_o_d
        return p_o_d

    def get_p_u_d(self, u, field_weights, doc_id):
        """
        p(u|d) = sum_{f in F} p(u|d_f) p(f|u) for unordered search

        :param u: phrase (unordered search)
        :param field_weights: Dictionary {f: p_f_u, ...}
        :param doc_id: entity id
        :return  p(o|d)
        """
        lucene_doc_id_t = self.lucene_term.get_lucene_document_id(doc_id)
        self.set_phrase_freq(self.UNORDERED, u, field_weights)
        p_u_d = 0
        for f, p_f_u in field_weights.iteritems():
            if self.DEBUG:
                print "\tu:", u, "f:", f
            tf_t_d_f = self.phrase_freq[self.UNORDERED][u].get(f, {}).get(doc_id, 0)
            tf_t_C_f = self.phrase_freq[self.UNORDERED][u].get(f, {}).get('coll_freq', 0)
            p_u_d_f = self.scorer_lm_term.get_term_prob(lucene_doc_id_t, f, u, tf_t_d_f=tf_t_d_f, tf_t_C_f=tf_t_C_f)
            p_u_d += p_u_d_f * p_f_u
            if self.DEBUG:
                print "\t\tp(u|d_f):", p_u_d_f, "p(f|u):", p_f_u, "p(u|d_f).p(f|u):", p_u_d_f * p_f_u
        if self.DEBUG:
            print "\tp(u|d):", p_u_d
        return p_u_d

    def get_p_e_d(self, e, field_weights, doc_id):
        """
        p(e|d) = sum_{f in F} p(e|d_f) p(f|e)

        :param e: entity URI
        :param field_weights: Dictionary {f: p_f_t, ...}
        :param doc_id: entity id
        :return p(e|d)
        """
        if self.DEBUG:
            print "\te:", e
        p_e_d = 0
        for f, p_f_e in field_weights.iteritems():
            p_e_d_f = self.__get_uri_prob(doc_id, f, e)
            p_e_d += p_e_d_f * p_f_e
            if self.DEBUG:
                print "\t\tp(e|d_f):", p_e_d_f, "p(f|e):", p_f_e, "p(e|d_f).p(f|e):", p_e_d_f * p_f_e
        if self.DEBUG:
            print "\tp(e|d):", p_e_d
        return p_e_d

    def __get_uri_prob(self, doc_id, field, e, lambd=0.1):
        """
        P(e|d_f) = P(e|d_f)= (1 - lambda) tf(e, d_f)+ lambda df(f, e) / df(f)

        :param doc_id: document id
        :param field: field name
        :param e: entity uri
        :param lambd: smoothing parameter
        :return: P(e|d_f)
        """
        if self.DEBUG:
            print "\t\tf:", field
        lucene_doc_id_u = self.lucene_uri.get_lucene_document_id(doc_id)
        tf = self.scorer_lm_uri.get_tf(lucene_doc_id_u, field)
        tf_e_d_f = 1 if tf.get(e, 0) > 0 else 0
        df_f_e = self.lucene_uri.get_doc_freq(e, field)
        df_f = self.lucene_uri.get_doc_count(field)
        p_e_d_f = ((1 - lambd) * tf_e_d_f) + (lambd * df_f_e / df_f)
        if self.DEBUG:
            print "\t\t\ttf(e,d_f):", tf_e_d_f, "df(f, e):", df_f_e, "df(f):", df_f, "P(e|d_f):", p_e_d_f
        return p_e_d_f


class ScorerFSDM(ScorerMRF):
    DEBUG_FSDM = 0

    def __init__(self, lucene_term, lucene_uri, params, query_annot):
        ScorerMRF.__init__(self, lucene_term, lucene_uri, params, query_annot)
        self.lambda_T = self.params['lambda'][0]
        self.lambda_O = self.params['lambda'][1]
        self.lambda_U = self.params['lambda'][2]
        self.T = self.query_annot.T

    def score_doc(self, doc_id):
        """    
        P(q|e) = lambda_T sum_{t in T}P(t|d) + lambda_O sum_{o in O}P(o|d) + lambda_U sum_{u in U}P(u|d)
        P(t|d) = sum_{f in F} p(t|d_f) p(f|t)
        P(o|d) = sum_{f in F} p(o|d_f) p(f|o)
        P(u|d) = sum_{f in F} p(u|d_f) p(f|u)

        :param doc_id: document id
        :return: p(q|d)
        """
        if self.DEBUG_FSDM:
            print "Scoring doc ID=" + doc_id

        if self.lucene_term.get_lucene_document_id(doc_id) is None:
            return None

        p_T_d = 0
        if self.lambda_T != 0:
            for t in self.T:
                p_t_d = self.get_p_t_d(t, self.get_field_weights(self.TERM, t), doc_id)
                if p_t_d != 0:
                    p_T_d += math.log(p_t_d)

        p_O_d = 0
        if self.lambda_O != 0:
            for b in self.bigrams:
                p_o_d = self.get_p_o_d(b, self.get_field_weights(self.ORDERED, b), doc_id)
                if p_o_d != 0:
                    p_O_d += math.log(p_o_d)

        p_U_d = 0
        if self.lambda_U != 0:
            for b in self.bigrams:
                p_u_d = self.get_p_u_d(b, self.get_field_weights(self.UNORDERED, b), doc_id)
                if p_u_d != 0:
                    p_U_d += math.log(p_u_d)

        p_q_d = (self.lambda_T * p_T_d) + (self.lambda_O * p_O_d) + (self.lambda_U * p_U_d)
        if self.DEBUG_FSDM:
            print "\t\tP(T|d) = ", p_T_d, "P(O|d):", p_O_d, "p(U|d):", p_U_d,  "P(q|d):", p_q_d

        return p_q_d


class ScorerELR(ScorerFSDM):
    DEBUG_ELR = 0

    def __init__(self, lucene_term, lucene_uri, params, query_annot):
        ScorerFSDM.__init__(self, lucene_term, lucene_uri, params, query_annot)
        self.lambda_E = self.params['lambda'][3]
        self.E = ScorerMRF.normalize_el_scores(self.query_annot.E)

    def score_doc(self, doc_id):
        """
        P(q|e) = lambda_T sum_{t}P(t|d) + lambda_O sum_{o}P(o|d) + lambda_U sum_{u}P(u|d) + + lambda_E sum_{e}P(e|d)
        P(T|d) = sum_{f in F} p(t|d_f) p(f|t)
        P(O|d) = sum_{f in F} p(o|d_f) p(f|o)
        P(U|d) = sum_{f in F} p(u|d_f) p(f|u)
        P(E|d) = sum_{f in F} p(e|d_f) p(f|e)

        :param doc_id: document id
        :return: p(q|d)
        """
        if self.DEBUG_ELR:
            print "Scoring doc ID=" + doc_id

        if self.lucene_term.get_lucene_document_id(doc_id) is None:
            # print doc_id,  self.lucene_term.get_lucene_document_id(doc_id)
            return None

        p_T_d = 0
        n_T = len(self.T)
        if self.lambda_T != 0:
            for t in self.T:
                p_t_d = self.get_p_t_d(t, self.get_field_weights(self.TERM, t), doc_id)
                if p_t_d != 0:
                    p_T_d += math.log(p_t_d) / n_T

        p_O_d = 0
        n_O = len(self.bigrams)
        if self.lambda_O != 0:
            for b in self.bigrams:
                p_o_d = self.get_p_o_d(b, self.get_field_weights(self.ORDERED, b), doc_id)
                if p_o_d != 0:
                    p_O_d += math.log(p_o_d) / n_O

        p_U_d = 0
        n_U = len(self.bigrams)
        if self.lambda_U != 0:
            for b in self.bigrams:
                p_u_d = self.get_p_u_d(b, self.get_field_weights(self.UNORDERED, b), doc_id)
                if p_u_d != 0:
                    p_U_d += math.log(p_u_d) / n_U

        p_E_d = 0
        if self.lambda_E != 0:
            for e, score in self.E.iteritems():
                p_e_d = self.get_p_e_d(e, self.get_field_weights(self.URI, e), doc_id)
                if p_e_d != 0:
                    p_E_d += score * math.log(p_e_d)

        p_q_d = (self.lambda_T * p_T_d) + (self.lambda_O * p_O_d) + (self.lambda_U * p_U_d) + (self.lambda_E * p_E_d)
        if self.DEBUG_ELR:
            print "\t\tP(T|d) = ", p_T_d, "P(O|d):", p_O_d, "p(U|d):", p_U_d, "p(E|d):", p_E_d,  "P(q|d):", p_q_d

        return p_q_d
