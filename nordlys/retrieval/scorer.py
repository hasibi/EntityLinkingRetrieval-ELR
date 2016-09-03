"""
Various retrieval models for scoring a individual document for a given query.

@author: Faegheh Hasibi (faegheh.hasibi@idi.ntnu.no)
@author: Krisztian Balog (krisztian.balog@uis.no)
"""

from __future__ import division
import math
from lucene_tools import Lucene


class Scorer(object):
    """Base scorer class."""

    SCORER_DEBUG = 0

    def __init__(self, lucene, query, params):
        self.lucene = lucene
        self.query = query
        self.params = params
        self.lucene.open_searcher()
        """
        @todo consider the field for analysis
        """
        # NOTE: The analyser might return terms that are not in the collection.
        # These terms are filtered out later in the score_doc functions.
        self.query_terms = lucene.analyze_query(self.query) if query is not None else None

    @staticmethod
    def get_scorer(model, lucene, query, params):
        """
        Returns Scorer object (Scorer factory).

        :param model: accepted values: lucene, lm or mlm
        :param lucene: Lucene object
        :param query: raw query (to be analyzed)
        :param params: dict with models parameters
        """
        if model == "lm":
            print "\tLM scoring ... "
            return ScorerLM(lucene, query, params)
        elif model == "mlm":
            print "\tMLM scoring ..."
            return ScorerMLM(lucene, query, params)
        elif model == "prms":
            print "\tPRMS scoring ..."
            return ScorerPRMS(lucene, query, params)
        else:
            raise Exception("Unknown model '" + model + "'")


class ScorerLM(Scorer):
    def __init__(self, lucene, query, params):
        super(ScorerLM, self).__init__(lucene, query, params)
        self.smoothing_method = params.get('smoothing_method', "jm").lower()
        if (self.smoothing_method != "jm") and (self.smoothing_method != "dirichlet"):
            raise Exception(self.params['smoothing_method'] + " smoothing method is not supported!")
        self.tf = {}

    @staticmethod
    def get_jm_prob(tf_t_d, len_d, tf_t_C, len_C, lambd):
        """
        Computes JM-smoothed probability
        p(t|theta_d) = [(1-lambda) tf(t, d)/|d|] + [lambda tf(t, C)/|C|]

        :param tf_t_d: tf(t,d)
        :param len_d: |d|
        :param tf_t_C: tf(t,C)
        :param len_C: |C| = \sum_{d \in C} |d|
        :param lambd: \lambda
        :return:
        """
        p_t_d = tf_t_d / len_d if len_d > 0 else 0
        p_t_C = tf_t_C / len_C if len_C > 0 else 0
        return (1 - lambd) * p_t_d + lambd * p_t_C

    @staticmethod
    def get_dirichlet_prob(tf_t_d, len_d, tf_t_C, len_C, mu):
        """
        Computes Dirichlet-smoothed probability
        P(t|theta_d) = [tf(t, d) + mu P(t|C)] / [|d| + mu]

        :param tf_t_d: tf(t,d)
        :param len_d: |d|
        :param tf_t_C: tf(t,C)
        :param len_C: |C| = \sum_{d \in C} |d|
        :param mu: \mu
        :return:
        """
        if mu == 0:  # i.e. field does not have any content in the collection
            return 0
        else:
            p_t_C = tf_t_C / len_C if len_C > 0 else 0
            return (tf_t_d + mu * p_t_C) / (len_d + mu)

    def get_tf(self, lucene_doc_id, field):
        if lucene_doc_id not in self.tf:
            self.tf[lucene_doc_id] = {}
        if field not in self.tf[lucene_doc_id]:
            self.tf[lucene_doc_id][field] = self.lucene.get_doc_termfreqs(lucene_doc_id, field)
        return self.tf[lucene_doc_id][field]

    def get_term_prob(self, lucene_doc_id, field, t, tf_t_d_f=None, tf_t_C_f=None):
        """
        Returns probability of a given term for the given field.

        :param lucene_doc_id: internal Lucene document ID
        :param field: entity field name, e.g. <dbo:abstract>
        :param t: term
        :return: P(t|d_f)
        """
        # Gets term freqs for field of document
        tf = {}
        if lucene_doc_id is not None:
            tf = self.get_tf(lucene_doc_id, field)

        len_d_f = sum(tf.values())
        len_C_f = self.lucene.get_coll_length(field)

        tf_t_d_f = tf.get(t, 0) if tf_t_d_f is None else tf_t_d_f
        tf_t_C_f = self.lucene.get_coll_termfreq(t, field) if tf_t_C_f is None else tf_t_C_f
        if self.SCORER_DEBUG:
            print "\t\tt=" + t + ", f=" + field
            print "\t\t\tDoc:  tf(t,f)=" + str(tf_t_d_f) + "\t|f|=" + str(len_d_f)
            print "\t\t\tColl: tf(t,f)=" + str(tf_t_C_f) + "\t|f|=" + str(len_C_f)

        # JM smoothing: p(t|theta_d_f) = [(1-lambda) tf(t, d_f)/|d_f|] + [lambda tf(t, C_f)/|C_f|]
        if self.smoothing_method == "jm":
            lambd = self.params.get('smoothing_param', 0.1)
            p_t_d_f = self.get_jm_prob(tf_t_d_f, len_d_f, tf_t_C_f, len_C_f, lambd)
            if self.SCORER_DEBUG:
                print "\t\t\tJM smoothing:"
                print "\t\t\tDoc:  p(t|theta_d_f)=", p_t_d_f
        # Dirichlet smoothing
        elif self.smoothing_method == "dirichlet":
            mu = self.params.get('smoothing_param', self.lucene.get_avg_len(field))
            p_t_d_f = self.get_dirichlet_prob(tf_t_d_f, len_d_f, tf_t_C_f, len_C_f, mu)
            if self.SCORER_DEBUG:
                print "\t\t\tDirichlet smoothing:"
                print "\t\t\tmu:", mu
                print "\t\t\tDoc:  p(t|theta_d_f)=", p_t_d_f
        return p_t_d_f

    def get_term_probs(self, lucene_doc_id, field):
        """
        Returns probability of all query terms for the given field.

        :param lucene_doc_id: internal Lucene document ID
        :param field: entity field name, e.g. <dbo:abstract>
        :return: dictionary of terms with their probabilities
        """
        p_t_theta_d_f = {}
        for t in set(self.query_terms):
            p_t_theta_d_f[t] = self.get_term_prob(lucene_doc_id, field, t)
        return p_t_theta_d_f

    def score_doc(self, doc_id, lucene_doc_id=None):
        """
        Scores the given document using LM.

        :param doc_id: document id
        :param lucene_doc_id: internal Lucene document ID
        :return float, LM score of document and query
        """
        if self.SCORER_DEBUG:
            print "Scoring doc ID=" + doc_id

        if lucene_doc_id is None:
            lucene_doc_id = self.lucene.get_lucene_document_id(doc_id)

        field = self.params.get('field', Lucene.FIELDNAME_CONTENTS)

        p_t_theta_d = self.get_term_probs(lucene_doc_id, field)
        if sum(p_t_theta_d.values()) == 0:  # none of query terms are in the field collection
            if self.SCORER_DEBUG:
                print "\t\tP(q|" + field + ") = None"
            return None
        # p(q|theta_d) = prod(p(t|theta_d)) ; we return log(p(q|theta_d))
        p_q_theta_d = 0
        for t in self.query_terms:
            # Skips the term if it is not in the field collection
            if p_t_theta_d[t] == 0:
                continue
            if self.SCORER_DEBUG:
                print "\t\tP(" + t + "|" + field + ") = " + str(p_t_theta_d[t])
            p_q_theta_d += math.log(p_t_theta_d[t])
        if self.SCORER_DEBUG:
            print "\tP(d|q)=" + str(p_q_theta_d)
        return p_q_theta_d


class ScorerMLM(ScorerLM):
    def __init__(self, lucene, query, params):
        super(ScorerMLM, self).__init__(lucene, query, params)

    def get_mlm_term_prob(self, lucene_doc_id, weights, t):
        """
        Returns MLM probability for the given term and field-weights.

        :param lucene_doc_id: internal Lucene document ID
        :param weights: dictionary, {field: weights, ...}
        :param t: term
        :return: P(t|theta_d)
        """
        # p(t|theta_d) = sum(mu_f * p(t|theta_d_f))
        p_t_theta_d = 0
        for f, mu_f in weights.iteritems():
            p_t_theta_d_f = self.get_term_prob(lucene_doc_id, f, t)
            p_t_theta_d += mu_f * p_t_theta_d_f
        if self.SCORER_DEBUG:
            print "\t\tP(t|theta_d)=" + str(p_t_theta_d)
        return p_t_theta_d

    def get_mlm_term_probs(self, lucene_doc_id, weights):
        """
        Returns probability of all query terms for the given field weights.

        :param lucene_doc_id: internal Lucene document ID
        :param weights: dictionary, {field: weights, ...}
        :return: dictionary of terms with their probabilities
        """
        p_t_theta_d = {}
        for t in set(self.query_terms):
            if self.SCORER_DEBUG:
                print "\tt=" + t
            p_t_theta_d[t] = self.get_mlm_term_prob(lucene_doc_id, weights, t)
        return p_t_theta_d

    def score_doc(self, doc_id, lucene_doc_id=None):
        """
        Scores the given document using MLM model.

        :param doc_id: document id
        :param lucene_doc_id: internal Lucene document ID
        :return float, MLM score of document and query
        """
        if self.SCORER_DEBUG:
            print "Scoring doc ID=" + doc_id

        if lucene_doc_id is None:
            lucene_doc_id = self.lucene.get_lucene_document_id(doc_id)

        weights = self.params['field_weights']

        p_t_theta_d = self.get_mlm_term_probs(lucene_doc_id, weights)
        # none of query terms are in the field collection
        if sum(p_t_theta_d.values()) == 0:
            if self.SCORER_DEBUG:
                print "\t\tP_mlm(q|theta_d) = None"
            return None
        # p(q|theta_d) = prod(p(t|theta_d)) ; we return log(p(q|theta_d))
        p_q_theta_d = 0
        for t in self.query_terms:
            if p_t_theta_d[t] == 0:
                continue
            if self.SCORER_DEBUG:
                print "\t\tP_mlm(" + t + "|theta_d) = " + str(p_t_theta_d[t])
            p_q_theta_d += math.log(p_t_theta_d[t])

        return p_q_theta_d


class ScorerPRMS(ScorerLM):
    def __init__(self, lucene, query, params):
        super(ScorerPRMS, self).__init__(lucene, query, params)
        self.fields = self.params['fields']
        self.total_field_freq = None
        self.mapping_probs = None

    def score_doc(self, doc_id, lucene_doc_id=None):
        """
        Scores the given document using PRMS model.

        :param doc_id: document id
        :param lucene_doc_id: internal Lucene document ID
        :return float, PRMS score of document and query
        """
        if self.SCORER_DEBUG:
            print "Scoring doc ID=" + doc_id

        if lucene_doc_id is None:
            lucene_doc_id = self.lucene.get_lucene_document_id(doc_id)

        # gets mapping probs: p(f|t)
        p_f_t = self.get_mapping_probs()

        # gets term probs: p(t|theta_d_f)
        p_t_theta_d_f = {}
        for field in self.fields:
            p_t_theta_d_f[field] = self.get_term_probs(lucene_doc_id, field)
        # none of query terms are in the field collection
        if sum([sum(p_t_theta_d_f[field].values()) for field in p_t_theta_d_f]) == 0:
            return None

        # p(q|theta_d) = prod(p(t|theta_d)) ; we return log(p(q|theta_d))
        p_q_theta_d = 0
        for t in self.query_terms:
            if self.SCORER_DEBUG:
                print "\tt=" + t
            # p(t|theta_d) = sum(p(f|t) * p(t|theta_d_f))
            p_t_theta_d = 0
            for f in self.fields:
                if f in p_f_t[t]:
                    p_t_theta_d += p_f_t[t][f] * p_t_theta_d_f[f][t]
                    if self.SCORER_DEBUG:
                        print "\t\t\tf=" + f + ", p(t|f)=" + str(p_f_t[t][f]) + "  P(t|theta_d,f)=" + str(p_t_theta_d_f[f][t])

            if p_t_theta_d == 0:
                continue
            p_q_theta_d += math.log(p_t_theta_d)
            if self.SCORER_DEBUG:
                print "\t\tP(t|theta_d)=" + str(p_t_theta_d)
        return p_q_theta_d

    def get_mapping_probs(self):
        """Gets (cached) mapping probabilities for all query terms."""
        if self.mapping_probs is None:
            self.mapping_probs = {}
            for t in set(self.query_terms):
                self.mapping_probs[t] = self.get_mapping_prob(t)
        return self.mapping_probs

    def get_mapping_prob(self, t, coll_termfreq_fields=None):
        """
        Computes PRMS field mapping probability.
            p(f|t) = P(t|f)P(f) / sum_f'(P(t|C_{f'_c})P(f'))

        :param t: str
        :param coll_termfreq_fields: {field: freq, ...}
        :return Dictionary {field: prms_prob, ...}
        """
        if coll_termfreq_fields is None:
            coll_termfreq_fields = {}
            for f in self.fields:
                coll_termfreq_fields[f] = self.lucene.get_coll_termfreq(t, f)

        # calculates numerators for all fields: P(t|f)P(f)
        numerators = {}
        for f in self.fields:
            p_t_f = coll_termfreq_fields[f] / self.lucene.get_coll_length(f)
            p_f = self.lucene.get_doc_count(f) / self.get_total_field_freq()
            p_f_t = p_t_f * p_f
            if p_f_t > 0:
                numerators[f] = p_f_t
            if self.SCORER_DEBUG:
                print "\tf= " + f, "t= " + t + " P(t|f)=" + str(p_t_f) + " P(f)=" + str(p_f)

        # calculates denominator: sum_f'(P(t|C_{f'_c})P(f'))
        denominator = sum(numerators.values())

        mapping_probs = {}
        if denominator > 0:  # if the term is present in the collection
            for f in numerators:
                mapping_probs[f] = numerators[f] / denominator
                if self.SCORER_DEBUG:
                    print "\t\tf= " + f + " t= " + t + " p(f|t)= " + str(numerators[f]) + "/" + str(sum(numerators.values())) + \
                          " = " + str(mapping_probs[f])
        return mapping_probs

    def get_total_field_freq(self):
        """Returns total occurrences of all fields"""
        if self.total_field_freq is None:
            total_field_freq = 0
            for f in self.fields:
                total_field_freq += self.lucene.get_doc_count(f)
            self.total_field_freq = total_field_freq
        return self.total_field_freq