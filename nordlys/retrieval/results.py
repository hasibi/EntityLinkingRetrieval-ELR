"""
Result list representation.

- for each hit it holds score and both internal and external doc_ids

@author: Krisztian Balog (krisztian.balog@uis.no)
"""

import operator


class RetrievalResults(object):
    """Class for storing retrieval scores for a given query."""
    def __init__(self):
        self.scores = {}
        # mapping from external to internal doc_ids -s
        self.doc_ids = {}

    def append(self, doc_id, score, doc_id_int=None):
        """Adds document to the result list"""
        self.scores[doc_id] = score
        if doc_id_int is not None:
            self.doc_ids[doc_id] = doc_id_int

    def increase(self, doc_id, score):
        """Increases the score of a document (adds it to the results list
        if it is not already there)"""
        if doc_id not in self.scores:
            self.scores[doc_id] = 0
        self.scores[doc_id] += score

    def num_docs(self):
        """Returns the number of documents in the result list."""
        return len(self.scores)

    def get_scores_sorted(self):
        """Returns all results sorted by score"""
        return sorted(self.scores.iteritems(), key=operator.itemgetter(1), reverse=True)

    def get_doc_id_int(self, doc_id):
        """Returns internal doc_id for a given doc_id."""
        if doc_id in self.doc_ids:
            return self.doc_ids[doc_id]
        return None

    def write_trec_format(self, query_id, run_id, out, max_rank=100):
        """Outputs results in TREC format"""
        rank = 1
        for doc_id, score in self.get_scores_sorted():
            if rank <= max_rank:
                out.write(query_id + "\tQ0\t" + doc_id + "\t" + str(rank) + "\t" + str(score) + "\t" + run_id + "\n")
            rank += 1
