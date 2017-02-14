"""
Creates a Lucene index for DBpedia from MongoDB.

- URI values are resolved using a simple heuristic
- fields are indexed as multi-valued
- catch-all fields are not indexed with positions, other fields are

--------------------------------------------------------------------------------------------------
NOTE: Please note that this code cannot be run due to dependencies to the DBpedia Mongo collection.
      Yet, this is the main code used fo generating the indices and can be used as a reference.
      To get the original indices, please contact the first author.
--------------------------------------------------------------------------------------------------

@author: Faegheh Hasibi (faegheh.hasibi@idi.ntnu.no)
@author: Krisztian Balog (krisztian.balog@uis.no)
"""

import sys
from urllib import unquote
from pprint import pprint

from nordlys import config
from nordlys.entity.config import COLLECTION_DBPEDIA
from nordlys.entity.dbpedia.fields import Fields
from nordlys.storage.mongo import Mongo
from nordlys.retrieval.lucene_tools import Lucene


class MongoDBToLucene(object):
    def __init__(self, host=config.MONGO_HOST, db=config.MONGO_DB, collection=COLLECTION_DBPEDIA):
        self.mongo = Mongo(host, db, collection)
        self.contents = None

    def __resolve_uri(self, uri):
        """Resolves the URI using a simple heuristic."""
        uri = unquote(uri)  # decode percent encoding
        if uri.startswith("<") and uri.endswith(">"):
            # Part between last ':' and '>', and _ replaced with space.
            # Works fine for <dbpedia:XXX> and <dbpedia:Category:YYY>
            return uri[uri.rfind(":") + 1:-1].replace("_", " ")
        else:
            return uri

    def __is_uri(self, value):
        """ Returns true if the value is uri. """
        if value.startswith("<dbpedia:") and value.endswith(">"):
            return True
        return False

    def __get_field_value(self, value, only_uris=False):
        """
        Converts mongoDB field value to indexable values by resolving URIs.
        It may be a string or a list and the return value is of the same data type.
        """
        if type(value) is list:
            nval = []  # holds resolved values
            for v in value:
                if not only_uris:
                    nval.append(Lucene.preprocess(self.__resolve_uri(v)))
                elif only_uris and self.__is_uri(v):
                    nval.append(v)
            return nval
        else:
            if not only_uris:
                return Lucene.preprocess(self.__resolve_uri(value))
            elif only_uris and self.__is_uri(value):
                return value
            # return self.__resolve_uri(value) if only_uris else value
        return None

    def __add_to_contents(self, field_name, field_value, field_type):
        """
        Adds field to document contents.
        Field value can be a list, where each item is added separately (i.e., the field is multi-valued).
        """
        if type(field_value) is list:
            for fv in field_value:
                self.__add_to_contents(field_name, fv, field_type)
        else:
            if len(field_value) > 0:  # ignore empty fields
                self.contents.append({'field_name': field_name,
                                      'field_value': field_value,
                                      'field_type': field_type})

    def build_index(self, index_config, only_uris=False, max_shingle_size=None):
        """Builds index.

        :param index_config: index configuration
        """
        lucene = Lucene(index_config['index_dir'], max_shingle_size)
        lucene.open_writer()  # generated shingle analyzer if the param is not None

        fieldtype_tv = Lucene.FIELDTYPE_ID_TV if only_uris else Lucene.FIELDTYPE_TEXT_TV
        fieldtype_tvp = Lucene.FIELDTYPE_ID_TV if only_uris else Lucene.FIELDTYPE_TEXT_TVP
        fieldtype_id = Lucene.FIELDTYPE_ID_TV if only_uris else Lucene.FIELDTYPE_ID
        fieldtype_ntv = Lucene.FIELDTYPE_ID_TV if only_uris else Lucene.FIELDTYPE_TEXT_NTV

        # iterate through MongoDB contents
        i = 0
        for mdoc in self.mongo.find_all():

            # this is just to speed up things a bit
            # we can skip the document right away if the ID does not start
            # with "<dbpedia:"
            if not mdoc[Mongo.ID_FIELD].startswith("<dbpedia:"):
                continue

            # get back document from mongo with keys and _id field unescaped
            doc = self.mongo.get_doc(mdoc)

            # check must_have fields
            skip_doc = False
            for f, v in index_config['fields'].iteritems():
                if ("must_have" in v) and (v['must_have']) and (f not in doc):
                    skip_doc = True
                    break

            if skip_doc:
                continue

            # doc contents is represented as a list of fields
            # (mind that fields are multi-valued)
            self.contents = []

            # each predicate to a separate field
            for f in doc:
                if f == Mongo.ID_FIELD:  # id is special
                    self.__add_to_contents(Lucene.FIELDNAME_ID, doc[f], fieldtype_id)
                if f in index_config['ignore']:
                    pass
                else:
                    # get resolved field value(s) -- note that it might be a list
                    field_value = self.__get_field_value(doc[f], only_uris)
                    # ignore empty fields
                    if (field_value is None) or (field_value == []):
                        continue

                    to_catchall_content = True if index_config['catchall_all'] else False

                    if f in index_config['fields']:
                        self.__add_to_contents(f, field_value, fieldtype_tvp)

                        # fields in index_config['fields'] are always added to catch-all content
                        to_catchall_content = True

                        # copy field value to other field(s)
                        # (copying is without term positions)
                        if "copy_to" in index_config['fields'][f]:
                            for f2 in index_config['fields'][f]['copy_to']:
                                self.__add_to_contents(f2, field_value, fieldtype_tv)

                    # copy field value to catch-all content field
                    # (copying is without term positions)
                    if to_catchall_content:
                        self.__add_to_contents(Lucene.FIELDNAME_CONTENTS, field_value, fieldtype_tv)

            # add document to index
            lucene.add_document(self.contents)

            i += 1
            if i % 1000 == 0:
                print str(i / 1000) + "K documents indexed"
        # close Lucene index
        lucene.close_writer()

        print "Finished indexing (" + str(i) + " documents in total)"


def main(argv):
    fields = {}
    top_fields = Fields().get_all()
    for f in top_fields:
        if f == "<rdfs:label>":
            fields[f] = {'must_have': True, 'copy_to': ["names"]}
        elif (f == "<foaf:name>") or (f == "!<dbo:wikiPageRedirects>"):
            fields[f] = {'copy_to': ["names"]}
        elif (f == "<rdf:type>") or (f == "<dcterms:subject>"):
            fields[f] = {'copy_to': ["types"]}
        elif f == "<rdfs:comment>":
            fields[f] = {'must_have': True}
        else:
            fields[f] = {}


    # Config of index7
    config_index7 = {'index_dir': "path/to/index",
                     'fields': fields,
                     'catchall_all': True,
                     'ignore': ["<owl:sameAs>"]  # except these
                    }

    # config of config7_only_uri; Similar to index7, but keeps only uris
    index_config7_only_uri = {'index_dir': "path/to/uri_only index",
                     'fields': fields,
                     'catchall_all': True,
                     'ignore': ["<owl:sameAs>"]  # except these
                    }

    pprint(config_index7)
    m2l = MongoDBToLucene()
    m2l.build_index(config_index7, only_uris=False)
    m2l.build_index(config_index7, only_uris=True)
    print "index build" + config_index7['index_dir']

if __name__ == "__main__":
    main(sys.argv[1:])
