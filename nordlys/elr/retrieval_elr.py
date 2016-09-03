"""
Class for entity retrieval

@author: Faegheh Hasibi (faegheh.hasibi@idi.ntnu.no)
"""

import argparse
import json
import os

from nordlys.config import QUERIES, TERM_INDEX_DIR, URI_INDEX_DIR, OUTPUT_DIR, ANNOTATIONS
from nordlys.elr.query_annot import QueryAnnot
from nordlys.elr.scorer_elr import ScorerMRF
from nordlys.retrieval.lucene_tools import Lucene
from nordlys.retrieval.results import RetrievalResults
from nordlys.retrieval.retrieval import Retrieval


class RetrievalELR(Retrieval):
    def __init__(self, model, query_file, annot_file, el_th=None, lambd=None, n_fields=None):
        query_file = query_file
        config = {'model': model,
                  'index_dir': TERM_INDEX_DIR,
                  'query_file': query_file,
                  'lambda': lambd,
                  'th': el_th,
                  'n_fields': n_fields,
                  'first_pass_num_docs': 1000,
                  'num_docs': 100,
                  'fields': None}

        lambd_str = "_lambda" + "_".join([str(l) for l in lambd]) if lambd is not None else ""
        th_str = "_th" + str(el_th) if el_th is not None else ""
        fields_str = str(n_fields) if n_fields is not None else ""
        run_id = model + fields_str + th_str + lambd_str
        config['run_id'] = run_id
        config['output_file'] = OUTPUT_DIR + "/" + run_id + ".treceval"
        super(RetrievalELR, self).__init__(config)

        self.annot_file = annot_file

    def _load_query_annotations(self):
        """Loads field annotation file."""
        self.query_annotations = json.load(open(self.annot_file))

    def _open_index(self):
        self.lucene_term = Lucene(TERM_INDEX_DIR)
        self.lucene_uri = Lucene(URI_INDEX_DIR)
        self.lucene_term.open_searcher()
        self.lucene_uri.open_searcher()

    def _close_index(self):
        self.lucene_term.close_reader()
        self.lucene_uri.close_reader()

    def _second_pass_scoring(self, res1, scorer):
        """
        Returns second-pass scoring of documents.

        :param res1: first pass results
        :param scorer: scorer object
        :return: RetrievalResults object
        """
        print "\tSecond pass scoring... "
        results = RetrievalResults()
        for doc_id, orig_score in res1.get_scores_sorted():
            score = scorer.score_doc(doc_id)
            results.append(doc_id, score)
        print "done"
        return results

    def retrieve(self, store_json=True):
        """Scores queries and outputs results."""
        self._open_index()
        self._load_queries()
        self._load_query_annotations()

        # init output file
        if os.path.exists(self.config['output_file']):
            os.remove(self.config['output_file'])
        out = open(self.config['output_file'], "w")
        print "Number of queries:", len(self.queries)

        for qid in sorted(self.queries):
            query = Lucene.preprocess(self.queries[qid])
            print "scoring [" + qid + "] " + query
            query_annot = QueryAnnot(self.query_annotations[qid], self.config['th'], qid=qid)

            # score documents
            res1 = self._first_pass_scoring(self.lucene_term, query)
            scorer = ScorerMRF.get_scorer(self.lucene_term, self.lucene_uri, self.config, query_annot)
            results = self._second_pass_scoring(res1, scorer)

            # write results to output file
            results.write_trec_format(qid, self.config['run_id'], out, self.config['num_docs'])
            break

        out.close()
        self._close_index()

        print "Output results: " + self.config['output_file']


def arg_parser():
    valid_models = ["lm", "mlm", "mlm-tc", "mlm-all", "prms", "sdm", "fsdm",
                    "lm_elr", "mlm_elr", "mlm-tc_elr", "prms_elr", "sdm_elr", "fsdm_elr"]
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Model name", type=str, choices=valid_models)
    parser.add_argument("-q", "--queries", help="Query file", type=str, default=QUERIES)
    parser.add_argument("-a", "--annot", help="Annotation file (with field mappings)", type=str, default=ANNOTATIONS)
    parser.add_argument("-t", "--threshold", help="Entity linking threshold", type=float, default=0.1)
    parser.add_argument("-n", "--nfields", help="number of fields", type=int, default=10)
    parser.add_argument("-l", "--lambd", help="Lambdas, comma separated values for ", type=str)
    args = parser.parse_args()
    return args


def main(args):
    lambda_params = None
    if args.lambd is not None:
        lambdas = args.lambd.split(",")
        lambda_params = [float(l.strip()) for l in lambdas]

    RetrievalELR(args.model, args.queries, args.annot, el_th=args.threshold, lambd=lambda_params,
                 n_fields=args.nfields).retrieve()

if __name__ == '__main__':
    main(arg_parser())
