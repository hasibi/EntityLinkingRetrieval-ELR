"""
Console application for general-purpose retrieval.

first pass: get top N documents using Lucene's default retrieval method (based on the catch-all content field)
second pass: perform (expensive) scoring of the top N documents using the Scorer class

General config parameters:
- index_dir: index directory
- query_file: query file (JSON)
- model: accepted values: lucene, lm, mlm, prms (default: lm)
- output_file: output file name
- output_format: (default: trec) -- not used yet
- run_id: run in (only for "trec" output format)
- num_docs: number of documents to return (default: 100)
- field_id: id field to be returned (default: Lucene.FIELDNAME_ID)
- first_pass_num_docs: number of documents in first-pass scoring (default: 10000)
- first_pass_field: field used in first pass retrieval (default: Lucene.FIELDNAME_CONTENTS)

Model-specific parameters:
- smoothing_method: jm or dirichlet (lm and mlm, default: jm)
- smoothing_param: value of lambda or alpha (jm default: 0.1, dirichlet default: average field length)
- field_weights: dict with fields and corresponding weights (only mlm)
- field: field name for LM model
- fields: fields for PRMS model


@author: Krisztian Balog (krisztian.balog@uis.no)
"""
from datetime import datetime

import sys
import json
import os
from nordlys.retrieval.lucene_tools import Lucene
from scorer import Scorer
from results import RetrievalResults


class Retrieval(object):
    def __init__(self, config):
        """
        Loads config file, checks params, and sets default values.

        :param config: JSON config file or a dictionary
        """
        # set configurations
        if type(config) == dict:
            self.config = config
        else:
            try:
                self.config = json.load(open(config))
            except Exception, e:
                print "Error loading config file: ", e
                sys.exit(1)

        # check params and set default values
        try:
            if 'index_dir' not in self.config:
                raise Exception("index_dir is missing")
            if 'query_file' not in self.config:
                raise Exception("query_file is missing")
            if 'output_file' not in self.config:
                raise Exception("output_file is missing")
            if 'run_id' not in self.config:
                raise Exception("run_id is missing")
            if 'model' not in self.config:
                self.config['model'] = "lm"
            if 'num_docs' not in self.config:
                self.config['num_docs'] = 100
            if 'field_id' not in self.config:
                self.config['field_id'] = Lucene.FIELDNAME_ID
            if 'first_pass_num_docs' not in self.config:
                self.config['first_pass_num_docs'] = 10000
            if 'first_pass_field' not in self.config:
                self.config['first_pass_field'] = Lucene.FIELDNAME_CONTENTS

            # model specific params
            if self.config['model'] == "lm" or self.config['model'] == "mlm" or self.config['model'] == "prms":
                if 'smoothing_method' not in self.config:
                    self.config['smoothing_method'] = "jm"
                # if 'smoothing_param' not in self.config:
                #     self.config['smoothing_param'] = 0.1

            if self.config['model'] == "mlm":
                if 'field_weights' not in self.config:
                    raise Exception("field_weights is missing")

            if self.config['model'] == "prms":
                if 'fields' not in self.config:
                    raise Exception("fields is missing")

        except Exception, e:
            print "Error in config file: ", e
            sys.exit(1)

    def _open_index(self):
        self.lucene = Lucene(self.config['index_dir'])

        self.lucene.open_searcher()

    def _close_index(self):
        self.lucene.close_reader()

    def _load_queries(self):
        self.queries = json.load(open(self.config['query_file']))

    def _first_pass_scoring(self, lucene, query):
        """
        Returns first-pass scoring of documents.

        :param query: raw query
        :return RetrievalResults object
        """
        print "\tFirst pass scoring... ",
        results = lucene.score_query(query, field_content=self.config['first_pass_field'],
                                     field_id=self.config['field_id'],
                                     num_docs=self.config['first_pass_num_docs'])
        print results.num_docs()
        return results

    def _second_pass_scoring(self, res1, scorer):
        """
        Returns second-pass scoring of documents.

        :param res1: first pass results
        :return: RetrievalResults object
        """
        print "\tSecond pass scoring... "
        results = RetrievalResults()
        for doc_id, orig_score in res1.get_scores_sorted():
            doc_id_int = res1.get_doc_id_int(doc_id)
            score = scorer.score_doc(doc_id, doc_id_int)
            results.append(doc_id, score)
        print "done"
        return results

    def retrieve(self):
        """Scores queries and outputs results."""
        s_t = datetime.now()  # start time
        total_time = 0.0

        self._load_queries()
        self._open_index()

        # init output file
        if os.path.exists(self.config['output_file']):
            os.remove(self.config['output_file'])
        out = open(self.config['output_file'], "w")

        for query_id in sorted(self.queries):
            # query = Query.preprocess(self.queries[query_id])
            query = Lucene.preprocess(self.queries[query_id])
            print "scoring [" + query_id + "] " + query
            # first pass scoring
            res1 = self._first_pass_scoring(self.lucene, query)
            # second pass scoring (if needed)
            if self.config['model'] == "lucene":
                results = res1
            else:
                scorer = Scorer.get_scorer(self.config['model'], self.lucene, query, self.config)
                results = self._second_pass_scoring(res1, scorer)
            # write results to output file
            results.write_trec_format(query_id, self.config['run_id'], out, self.config['num_docs'])

        # close output file
        out.close()
        # close index
        self._close_index()

        e_t = datetime.now()  # end time
        diff = e_t - s_t
        total_time += diff.total_seconds()
        time_log = "Execution time(sec):\t" + str(total_time) + "\n"
        print time_log


def print_usage():
    print sys.argv[0] + " <config_file>"
    sys.exit()


def main(argv):
    if len(argv) < 1:
        print_usage()

    r = Retrieval(argv[0])
    r.retrieve()


if __name__ == '__main__':
    main(sys.argv[1:])
