from os.path import join, basename, exists
from os import makedirs, rename
import os
from pathlib import Path
import pandas as pd
import logging
import sys
import h5py
# import dask.dataframe as dd
import inspect

from lire.log_tools import logger

import json
# from lire.data_tools import data_loader
from lire.data_tools import data_downloader
from lire.data_tools import data_compress
from lire.data_tools import data_reader
from lire.data_tools.continual_routine import ContinualDataset
class MSMarcoDataset(object):
    def __init__(self, storage="full", chunk_size="100MB"):
        self.storage = storage
        self.chunk_size = chunk_size

    @staticmethod
    def _load_pandas_dataframe(filepath, sep='\t', nrows=None):
        dataframe = pd.read_csv(filepath, sep=sep, header=None, index_col=0, nrows=nrows)
        return dataframe

    @staticmethod
    def _load_dask_dataframe(filepath, sep='\t', nrows=None):
        ''' Loading a dataframe using dask api for chunk loading.

            This is much memory efficient than pandas, however it is 
            longer to access data. We advise using fast storage to works
            efficiently with this type of loading.

        '''
        dataframe = dd.read_csv(filepath, sep=sep, header=None,
                                          index_col=0, encoding='utf8',
                                          blocksize=chunk_size)
        
        return dataframe

    def _load_dataframe(self, filepath, sep='\t', nrows=None):
        if(self.storage == 'chunk'):
            path = Path(filepath)
            filename = path.stem
            directory = str(path.parent)
            chunk_directory = join(directory, filename+"_parquet")

            if(not exists(chunk_directory)):
                df = dd.read_csv(filepath, sep=sep, header=None, 
                                           blocksize=self.chunk_size).set_index(0)
                df.repartition(partition_size = self.chunk_size)
                df.to_parquet(chunk_directory)
            
        else:
            return self._load_pandas_dataframe(filepath, sep=sep, nrows=nrows)
    
    @classmethod
    def _download_files(cls, root_folder, key_data, configuration_path):
        config = logger.ConfigurationFile(configuration_path).content
        print(config)
        foldername = config["folder"]["foldername"]


        folder_path = join(root_folder, foldername)
        makedirs(folder_path, exist_ok=True)

        download_url = [(key_val, url, basename(url)) for key_val, url in config["download"][key_data].items()]
        download_url +=  [(key_val, url, basename(url)) for key_val, url in config["download"]["common"].items()]

        for download_item in download_url:
            key_val, url, filename = download_item
            if(not exists(join(folder_path, key_val+".data"))):

                print('Download "', filename, '" at "',
                    url, '" to "', join(folder_path, filename),'"')
                data_downloader.download_to(url, join(folder_path, filename))
                try:
                    data_compress.untar(join(folder_path, filename), join(folder_path, key_val+".data")) 
                except Exception:
                    # TODO : check not a compressed file
                    rename(join(folder_path, filename), join(folder_path, key_val+".data"))
            else : 
                print('Already downloaded "', filename, '" at "',
                    url, '" to "', join(folder_path, filename),'"')
        return {key_val:join(folder_path, filename)
                for key_val, url, filename in download_url}

class MSMarcoRankingDataset(MSMarcoDataset):

    configuration_path = "dataset_info/MSMarcoRankingDataset.json"


    def __init__(self, folder, download=False, split="dev", force=False, subset=None):
        super(MSMarcoRankingDataset).__init__()
        self.folder = folder
        self.download = download
        self.split = split

        self.configuration_path = join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),self.configuration_path)

        self.configuration = logger.ConfigurationFile(self.configuration_path)
        self.subset = subset
        try: 
            # loading from local
            raise Exception()
        except Exception:
            self._download_files(self.folder, self.split)
            if(not self.download):
                raise Exception("No dataset at ", self.folder, ", use download=True, to download the corpus")
    
    @staticmethod
    def _load_collection(filepath, sep='\t'):
        dataframe = pd.read_csv(filepath, sep=sep)
        return dataframe



    def _load_from_file(self):
        self.documents_collection = cls._load_collection(os.path.join(self.folder, cls.documents_collection))
        self.queries_collection = cls._load_collection(os.path.join(self.folder, cls.queries_collection))
        self.queries_documents_top100 = cls._load_collection(os.path.join(self.folder, cls.queries_documents), sep="\s")


    def get_documents_related_to_a_query(self, index):
        query_str, query_index = self.queries_collection.get[i]
        documents_id = self.queries_documents_top100.get(query_index)
        documents = [(self.documents_collection.get(id)[1]) for id in documents_id]
        return query_str, documents

class MSMarcoPassageRankingDataset(MSMarcoDataset):

    configuration_path = "dataset_info/MSMarcoPassageRankingDataset.json"


    def __init__(self, folder, download=False, split="train",
                 force=False, storage="full", getter="positive",
                 subset_size=None, load_data=True):
        super(MSMarcoPassageRankingDataset, self).__init__(storage=storage)
        self.folder = folder
        self.download = download
        self.split = split
        self.getter = getter
        self.configuration_path =\
            join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),self.configuration_path)
        self.configuration = logger.ConfigurationFile(self.configuration_path)
        self.subset_size = subset_size

        self.document_transformation = lambda x: x
        self.query_transformation = lambda x: x
        self.output_transformation = lambda x: x
        self.documents = None
        self.queries = None
        self.qrels = None

        if(load_data):
            self._load_data()


    def _load_data(self):
        try:
            self._load_from_file()
        except Exception:
            if(not self.download):
                print(sys.exc_info()[0])
                raise Exception("No dataset at ", self.folder,
                                ", use download=True, to download the corpus")
            self._download_files(self.folder, self.split,
                                 self.configuration_path)
            self._load_from_file()

    def set_common_data(self, documents=None, queries=None, qrels=None):
        '''Set common data reference.
           The objective of this method is to avoid
           duplicate memory on common data
        '''
        if(documents is not None):
            self.documents = documents
        if(queries is not None):
            self.queries = queries
        if(qrels is not None):
            self.qrels = qrels

    @classmethod
    def load_queries(cls, split, folder):
        configuration_path = join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),
                                  cls.configuration_path)
        configuration = logger.ConfigurationFile(configuration_path)
        root_path = join(folder, configuration['folder']['foldername'])
        queries =\
            MSMarcoDataset._load_pandas_dataframe(join(root_path, split+"-queries.data"))
        return queries
    
    @classmethod
    def load_qrels(cls, split, folder):
        configuration_path = join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),
                                  cls.configuration_path)
        configuration = logger.ConfigurationFile(configuration_path)
        root_path = join(folder, configuration['folder']['foldername'])
        qrels =\
            data_reader.Qrels(filepath=join(root_path, split+"-qrels.data"),
                              getter_expression="qi -> q, dr")
        return qrels

    def get_nb_queries(self):
        return len(self.queries)

    def get_nb_annotated_queries(self):
        return len(self.qrels)

    def _load_from_file(self):
        root_path = join(self.folder, self.configuration['folder']['foldername'])
        # loading content
        if(self.documents is None):
            self.documents =\
                self._load_dataframe(join(root_path, "documents.data"))
        if(self.queries is None):
            self.queries =\
                self._load_dataframe(join(root_path, self.split+"-queries.data"))
        # loading relations
        if(self.qrels is None):
            self.qrels =\
                data_reader.Qrels(filepath=join(root_path, self.split+"-qrels.data"), getter_expression="qi -> q, dr")
        if(self.split == "train" and self.getter == "triplet"):
            self.query_positive_negative =\
                self._load_dataframe(join(root_path, "train_positive_negative.data"), 
                                     nrows=self.subset_size)

        if(self.split == "dev" and self.getter == "top1000"):
            with open(join(root_path, "dev_top_1000.json")) as f:
                self.top_1000_json = json.load(f)
                if(self.subset_size is None):
                    self.keys_top_1000 = list(self.top_1000_json.keys())
                else:
                    self.keys_top_1000 = list(self.top_1000_json.keys())[:self.subset_size]
                self.top_1000 = [[int(k), v] for k in self.keys_top_1000 
                                 for v in self.top_1000_json[k]]

    def get_query_positive_negative(self, index):

        index_row = self.query_positive_negative.iloc[index] 
        query_id, positive_document_id, negative_document_id =\
            index_row.name, index_row[1], index_row[2]
        query_str = self.queries.loc[query_id][1]
        positive_str = self.documents.loc[positive_document_id][1]
        negative_str = self.documents.loc[negative_document_id][1]
        
        return (query_str, positive_str, negative_str)

    def set_document_transform(self, transformation):
        self.document_transformation = transformation

    def set_query_transform(self, transformation):
        self.query_transformation = transformation

    def set_output_transformation(self, transformation):
        self.output_transformation = transformation


    def __getitem__(self, index):
        if(self.getter == "positive"):
            query_id, documents_id = self.qrels[index]
            query_tr = self.query_transformation(self.queries.loc[int(query_id)][1])
            documents_tr = [self.document_transformation(self.documents.loc[int(document_id)][1]) for document_id in documents_id]
            return self.output_transformation((query_tr, documents_tr))

        if(self.getter == "triplet"):
            q, p, n = self.get_query_positive_negative(index)
            return self.output_transformation((self.query_transformation(q),
                                               self.document_transformation(p),
                                               self.document_transformation(n)))
        if(self.getter == "top1000"):
            query_doc = self.top_1000[index]
            query_id = query_doc[0]
            document_id = query_doc[1]
            return query_id, document_id,\
                   self.output_transformation((self.query_transformation(self.queries.loc[query_id][1]),
                                               self.document_transformation(self.documents.loc[
                                                   document_id][1])))

    def __len__(self):
        if(self.getter == "positive"):
            return len(self.qrels)
        elif(self.getter == "top1000"):
            return len(self.top_1000)
        else:
            return len(self.query_positive_negative)


class MSMarcoPassageRankingTopicQueryDataset(MSMarcoPassageRankingDataset, ContinualDataset):
    configuration_path = "dataset_info/MSMarcoPassageRankingTopicQueryDataset.json"

    def __init__(self, *args, **kwargs):
        MSMarcoPassageRankingDataset.__init__(self, *args, **kwargs)
        ContinualDataset.__init__(self)

    def _load_from_file(self):
        root_path = join(self.folder, self.configuration['folder']['foldername'])
        # loading content
        self.documents =\
            self._load_dataframe(join(root_path, "documents.data"))
        self.queries =\
             self._load_dataframe(join(root_path, self.split+"-queries.data"))
        # loading relations
        self.qrels =\
            data_reader.Qrels(filepath=join(root_path, self.split+"-qrels.data"), getter_expression="qi -> q, dr")
        if(self.split == "train" and self.getter == "triplet"):
            self.h5_split = h5py.File(os.path.join(root_path, "hdf5-data.data"), "r")


        with open(join(root_path, self.split+"-topics.data")) as f:
            self.topics = json.load(f)
        self.qrels.set_getter("q -> q, dr")
        print("Creating the index")
        self.set_current_task_by_id(0)


    def set_current_task_by_id(self, task_id):
        super(MSMarcoPassageRankingTopicQueryDataset, self).set_current_task_by_id(task_id)
        if(self.getter == "triplet" and self.split == "train"):
            self.current_task_index =\
                self.h5_split["/train/qid-pid-nid-cluster"][str(task_id)]

        if(self.getter == "positive"):
            self.current_task_index =\
                self.topics[task_id]

        # if(self.getter == "top1000"):
        #     top_1000 = self.h5_split["/"+self.split+"/"+self.split+"-top-1000.data"]
        #     data = []
        #     cumsize = []
        #     for k, v in self.topics[task_id].items():
        #         try:
        #             rel_doc = top_1000[str(k)
        #             data.append(k, rel_doc ])
        #             cumsize.append[len(rel_doc)]
                    
        #         except Exception:
        #             print(str(k) + " not in set")
            
                

                    
            

    def get_query_positive_negative(self, index):
        query_id, positive_document_id, negative_document_id =\
            self.current_task_index[index]
        query_str = self.queries.loc[query_id][1]
        positive_str = self.documents.loc[positive_document_id][1]
        negative_str = self.documents.loc[negative_document_id][1]
        
        return (query_str, positive_str, negative_str)

    def __getitem__(self, index):
        if(self.getter == "positive"):
            query_id, documents_id = self.qrels[str(self.current_task_index[index])]
            query_tr = self.query_transformation(self.queries.loc[int(query_id)][1])
            documents_tr = [self.document_transformation(self.documents.loc[int(document_id)][1]) for document_id in documents_id]
            return self.output_transformation((query_tr, documents_tr))

        if(self.getter == "triplet"):
            q, p, n = self.get_query_positive_negative(index)
            return self.output_transformation((self.query_transformation(q),
                    self.document_transformation(p), 
                    self.document_transformation(n)))
    
    def get_nb_queries(self):
        return len(self.topics[self.get_current_task_id()])

    def get_nb_tasks(self):
        return len(self.topics)

    def __len__(self):
        if(self.getter == "positive"):
            return len(self.current_task_index)
        else:
            return len(self.current_task_index)


class MSMarcoPassageRankingTopicQueryCleanedDataset(MSMarcoPassageRankingTopicQueryDataset):
    configuration_path = "dataset_info/MSMarcoPassageRankingTopicQueryCleanedDataset.json"
    def __init__(self, *args, **kwargs):
        super(MSMarcoPassageRankingTopicQueryCleanedDataset, self).__init__(*args, **kwargs)

class MSMarcoPassageRankingTopicSmall(MSMarcoPassageRankingTopicQueryDataset):
    configuration_path = "dataset_info/MSMarcoPassageRankingTopicSmall.json"
    def __init__(self, *args, **kwargs):
        super(MSMarcoPassageRankingTopicSmall, self).__init__(*args, **kwargs)