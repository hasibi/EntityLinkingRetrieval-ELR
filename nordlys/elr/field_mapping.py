"""
Computes PRMS field mapping probabilities.

@author: Faegheh Hasibi (faegheh.hasibi@idi.ntnu.no)
"""

from __future__ import division
from pprint import PrettyPrinter

from nordlys.retrieval.scorer import ScorerPRMS
from nordlys.elr.top_fields import TopFields


class FieldMapping(object):
    DEBUG = 0
    MAPPING_DEBUG = 0

    def __init__(self, lucene_term, lucene_uri, n):
        self.lucene_term = lucene_term
        self.lucene_uri = lucene_uri
        self.n = n

    def map(self, query_annot, slop=None):
        """
        Computes PRMS field mapping probabilities for URIs, terms, ordered, and unordered phrases.

        :param query_annot: nordlys.elr.QueryAnnot
        :param slop: number of terms in between
        :return: interprets: {'uris': {uri:{field: prob, ..}, ..}, 'terms': {..}, 'ordered': {..}, 'unordered': {..}}
        """
        T, phrases, E = set(query_annot.T), set(query_annot.get_all_phrases()), set(query_annot.E.keys())
        field_mappings = {'uris': self.get_mapping_uris(E),
                          'terms': self.get_mapping_terms(T),
                          'ordered': self.get_mapping_phrases(phrases, 0, True)}
        print "  ordered done!"
        if slop is not None:
            field_mappings['unordered'] = self.get_mapping_phrases(phrases, slop, False)
            print "  unordered done!"
        print "==="
        return field_mappings

    def get_mapping_uris(self, uris):
        """
        Computes field mapping probability for URIs.

        :param uris: list of uris
        :return: Dictionary {uri: {field: weight, ..}, ..}
        """
        field_mappings = {}
        for uri in uris:
            top_fields = TopFields(self.lucene_uri).get_top_term(uri, self.n)
            scorer_prms = ScorerPRMS(self.lucene_uri, None, {'fields': top_fields})
            field_mappings[uri] = scorer_prms.get_mapping_prob(uri)
            if self.DEBUG:
                print uri
                PrettyPrinter(depth=4).pprint(sorted(field_mappings[uri].items(), key=lambda f: f[1], reverse=True))
        return field_mappings

    def get_mapping_terms(self, terms):
        """
        Computes PRMS field mapping probability for terms.

        :param terms: list of terms
        :return: Dictionary {term: {field: weight, ..}, ..}
        """
        field_mappings = {}
        top_fields = TopFields(self.lucene_term).get_top_index(self.n)
        for term in terms:
            scorer_prms = ScorerPRMS(self.lucene_term, None, {'fields': top_fields})
            field_mappings[term] = scorer_prms.get_mapping_prob(term)
            if self.DEBUG:
                print term
                PrettyPrinter(depth=4).pprint(sorted(field_mappings[term].items(), key=lambda f: f[1], reverse=True))
        return field_mappings

    def get_mapping_phrases(self, phrases, slop, ordered):
        """
        Computes PRMS field mapping probability for phrases.

        :param phrases: list of phrases
        :param ordered: if True, performs ordered search
        :param slop: number of terms between the terms of phrase
        :return: Dictionary {phrase: {field: weight, ..}, ..}
        """
        field_mappings = {}
        top_fields = TopFields(self.lucene_term).get_top_index(self.n)
        for phrase in phrases:
            coll_freqs = self.__get_coll_freqs(phrase, top_fields, slop, ordered)
            scorer_prms = ScorerPRMS(self.lucene_term, None, {'fields': top_fields})
            field_mappings[phrase] = scorer_prms.get_mapping_prob(phrase, coll_termfreq_fields=coll_freqs)
            if self.DEBUG:
                print phrase
                PrettyPrinter(depth=4).pprint(sorted(field_mappings[phrase].items(), key=lambda f: f[1], reverse=True))
        return field_mappings

    def __get_coll_freqs(self, phrase, fields, slop, ordered):
        """Gets collection term frequency for all fields."""
        coll_freqs = {}
        for f in fields:
            doc_phrase_freq = self.lucene_term.get_doc_phrase_freq(phrase, f, slop=slop, ordered=ordered)
            coll_freqs[f] = sum(doc_phrase_freq.values())
        return coll_freqs