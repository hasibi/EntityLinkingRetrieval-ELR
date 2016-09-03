"""
This class returns top fields based on document frequency

@author: Faegheh Hasibi (faegheh.hasibi@idi.ntnu.no)
"""

from nordlys.retrieval.lucene_tools import Lucene


class TopFields(object):
    DEBUG = 0

    def __init__(self, lucene):
        self.lucene = lucene
        self.__fields = None

    @property
    def fields(self):
        if self.__fields is None:
            self.__fields = set(self.lucene.get_fields())
        return self.__fields

    def get_top_index(self, n):
        """Return top-n fields with highest document frequency across the whole index"""
        doc_freq_field = {}
        for field in self.fields:
            if field == Lucene.FIELDNAME_ID:
                continue
            doc_freq_field[field] = self.lucene.get_doc_count(field)
        return self.__get_top_n(doc_freq_field, n)

    def get_top_term(self, term, n):
        """Returns top-n fields with highest document frequency for the given term."""
        doc_freq = {}
        if self.DEBUG:
            print "Term:[" + term + "]"
        for field in self.fields:
            df = self.lucene.get_doc_freq(term, field)
            if df > 0:
                doc_freq[field] = df
        top_fields = self.__get_top_n(doc_freq, n)
        return top_fields

    def __get_top_n(self, fields_freq, n):
        """Sorts fields and returns top-n."""
        sorted_fields = sorted(fields_freq.items(), key=lambda item: (item[1], item[0]), reverse=True)
        top_fields = dict()
        i = 0
        for field, freq in sorted_fields:
            if i >= n:
                break
            i += 1
            top_fields[field] = freq
            if self.DEBUG:
                print "(" + field + ", " + str(freq) + ")",
        if self.DEBUG:
            print "\nNumber of fields:", len(top_fields), "\n"
        return top_fields
