"""
Global nordlys config.

@author: Faegheh Hasibi (faegheh.hasibi@idi.ntnu.no)
@author: Krisztian Balog (krisztian.balog@uis.no)
"""

from os import path

NORDLYS_DIR = path.dirname(path.abspath(__file__))
DATA_DIR = path.dirname(path.dirname(path.abspath(__file__))) + "/data"
OUTPUT_DIR = path.dirname(path.dirname(path.abspath(__file__))) + "/runs"

TERM_INDEX_DIR = "path/to/term/index"
URI_INDEX_DIR = "path/to/URI/index"
print "Term index:", TERM_INDEX_DIR
print "URI index:", URI_INDEX_DIR

QUERIES = DATA_DIR + "/queries.json"
ANNOTATIONS = DATA_DIR + "/tagme_annotations.json"

