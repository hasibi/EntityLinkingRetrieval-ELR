# ELR: Exploiting Entity Linking in Queries for Entity Retrieval 

This repository contains resources developed within the following paper:

	F. Hasibi, K. Balog, and S.E. Bratsberg. “Exploiting Entity Linking in Queries for Entity Retrieval”,
	In proceedings of ACM SIGIR International Conference on the Theory of Information Retrieval (ICTIR ’16), Newark, DE, USA, Sep 2016.


This repository is structured as follows:

- `nordlys/`: Code required for running entity retrieval methods.
- `data/`: Query set and data required for running the code.
- `qrels/`: Qrel files for [DBpedia-entity test collection](http://krisztianbalog.com/resources/sigir-2013-dbpedia/) (version 3.9).
- `runs/`: Run files reported in the paper.


## Usage

Use the follwoing command to run the code:

```
python -m nordlys.elr.retrieval_elr <model_name>
```
Using this command, the retrieval results are computed based on the recommended parameters in the paper. 
For detailed descriptions and setting different parameters read the help using this command `python -m nordlys.elr.retrieval_elr -h`.

## Code

You can read the `nordlys/elr/scorer_elr.py` for the actual implementation of the ELR framework and the baseline methods.

## Data

The indices required for running this code are described in the paper. You can also contact the authors to get the indices.
The follwoing files under the `data` folder are also required for running the code:

- `queries.json`: The DBpedia-entity queries, stopped as described in the paper.
- `tagme_annotations.json`: Entity annotaitons of the queries obtained from the [TAGME API](https://tagme.d4science.org/tagme/).



## Contact

If you have any questions, feel free to contact Faegheh Hasibi at <faegheh.hasibi@idi.ntnu.no>.
