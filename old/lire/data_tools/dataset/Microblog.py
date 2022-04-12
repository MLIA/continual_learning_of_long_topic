from os.path import join, basename, exists
from os import makedirs, rename
import os
from pathlib import Path
import pandas as pd
# import dask.dataframe as dd
import inspect

from lire.log_tools import logger

# from lire.data_tools import data_loader
from lire.data_tools import data_downloader
from lire.data_tools import data_compress
from lire.data_tools import data_reader

class MicroblogRankingDataset(object):

    configuration_path = "dataset_info/MicroblogRankingDataset.json"

    def __init__(self, storage="full", chunk_size="100MB"):
        self.storage = storage
        self.chunk_size = chunk_size
        
        self.folder = folder
        self.download = download
        self.split = split
        self.getter = getter
        self.configuration_path = join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),self.configuration_path)

        self.configuration = logger.ConfigurationFile(self.configuration_path)
        self.subset_size = subset_size
        try: 
            self._load_from_file()
        except Exception:

            self._download_files(self.folder, self.split, self.configuration_path)
            if(not self.download):
                raise Exception("No dataset at ", self.folder, ", use download=True, to download the corpus")
        self.document_transformation = lambda x: x 
        self.query_transformation = lambda x: x
        self.output_transformation = lambda x: x

    @classmethod
    def _download_files(cls, root_folder, key_data, configuration_path):
        config = logger.ConfigurationFile(configuration_path).content
        print(config)
        foldername = config["folder"]["foldername"]


        folder_path = join(root_folder, foldername)
        makedirs(folder_path, exist_ok=True)

        download_url +=  [(key_val, url, basename(url)) for key_val, url in config["download"]["common"].items()]

        for download_item in download_url:
            key_val, url, filename = download_item
            if(not exists(join(folder_path, key_val+".data"))):
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


    def _load_from_file(self):
        root_path = join(self.folder, self.configuration['folder']['foldername'])
        raise Exception("Ex")
        # loading content
