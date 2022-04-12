import h5py
import tqdm
import sys
import numpy as np
from itertools import islice

class CSVReader():
    @classmethod
    def from_csv_to_dictionary(cls, csv_filepath, split='\t', col_name=None,
                               group_by_key=None, col_type=None, batch_size=int(1e4),
                               group_by_func="identity", head=None):
        nb_lines = 0
        csv_dictionary = {}
        store_key = True
        if(group_by_func == "identity"):
            store_key = False
            group_by_func = lambda x: x

        group_by_key = str(group_by_key)

        with open(csv_filepath, 'r') as csv_file:
            for lines in iter(lambda: tuple(islice(csv_file, batch_size)), ()):
                
                if(nb_lines == 0):
                    # checking coherence of data with given parameters
                    col_count = len(lines[0].split(split))
                    if(col_name is not None and col_count < len(col_name)):
                        raise Exception("Number of column is lower than indecated")

                    if(col_name is None):
                        col_name = [str(i) for i in range(col_count)]
                    

                    if(group_by_key is not None):
                        col_key_index = [i for i, name in enumerate(col_name)
                                            if(name == group_by_key)][0]
                        col_key_selector = lambda x : x[col_key_index]

                    if(col_type is None):
                        col_type = int

                    if(not isinstance(col_type, list)):
                        col_type = [col_type for col  in col_name]
                    col_transform = lambda seq_list: [t(x) for x, t in zip(seq_list, col_type)]

                if (head is not None and nb_lines + len(lines) > head):
                    if head - nb_lines == 0:
                        break
                    lines = lines[: head - nb_lines]
                nb_lines += len(lines)

                sys.stdout.flush()
                for line in lines:
                    line_list = line.split(split, len(col_name))
                    data_list = col_transform(line_list)
                    if(group_by_key is not None):

                        key = group_by_func(col_key_selector(data_list))
                        if(not store_key):
                            data_list = data_list[:col_key_index] +\
                                data_list[col_key_index + 1:]
                        if(len(data_list) == 0):
                            raise Exception("At least to columns is needed")
                        if(len(data_list) == 1):
                            data_list = data_list[0]
                        if key not in csv_dictionary:
                            csv_dictionary[key] = []
                        csv_dictionary[key].append(data_list)
                    else:
                        csv_dictionary[len(dict)] = data_list
                print('Nb lines processed: ' + str(nb_lines), end='\r')
                if(head is not None and nb_lines >= head):
                    return csv_dictionary
        return csv_dictionary

class HDF5DatasetManager():
    type_str = {int:'i', float:'f'}

    @classmethod
    def batched_csv_to_hdf5(cls, hdf5_file, group_path : str, csv_filepath : str, 
                            split='\t', col_name=None, group_by_key=None,
                            col_type=None, batch_size=int(1e6), group_by_func=lambda x: x,
                            dataset_compression=None, head=None
                            ):
        ''' Write csv to dataset HDF5 using batch.
        The objective of the current function is to provide
        efficient way of writing HDF5 dataset from a csv source
        file. Mainly avoiding storing in memory all the csv content
        which would lead to full all the ram. User should provide 
        a batch size, a number of lines that can fit in memory. More
        the batch is large more the writting will be time efficient.

        Attributes
        ----------
        hdf5_file : str
            filepath of the hdf5 file.
        group_path : str
            the hierarchy of hdf5 group to store dataset(s) '/train/train_data/'.
        csv_filepath : str
            the filepath of the csv file to read and copy to hdf5.
        split : str
            the split token for csv file (for csv use ',' token, tsv '\t' token).
        col_name : list<str>
            the name of the columns, if not given store all the columns, if the 
            number of colums is lower than those contained in the csv file
            use only the columns given (or first n columns).
        group_by_key : str
            the columns that will be used for grouping, thus the name of the 
            dataset will be the key, if not given store all in 'data' named dataset.
        col_type : type or list<type>
            the type of the columns by default int (only float and int is currently 
            implemented).
        batch_size : int
            the size of the batch lines stored in memory (default=1e6).
        group_by_func : func
            a function applied on the key given, it transform the key into
            an other key, can merge keys.
        dataset_compression : str
            which type of compression used to store dataset. By default no compression
            is used. Can be "gzip", "lzf", "szip", for futher details refer to 
            https://docs.h5py.org/en/stable/high/dataset.html.
        '''
        data_group = h5py.File(hdf5_file, 'a')[group_path]
        nb_lines = 0
        key_count = {}
        print("Preprocessing and counting nb lines, can be long")
        with open(csv_filepath, 'r') as csv_file:
            for lines in iter(lambda: tuple(islice(csv_file, batch_size)), ()):
                if(nb_lines == 0):
                    # checking coherence of data with given parameters
                    col_count = len(lines[0].split(split))
                    if(col_name is not None and col_count < len(col_name)):
                        raise Exception("Number of column is lower than indecated")
                    
                    if(col_name is not None):
                        if(group_by_key is not None):
                            col_key_index = [i for i, name in enumerate(col_name)
                                             if(name == group_by_key)][0]
                            col_key_selector = lambda x : x[col_key_index]

                    if(col_name is None):
                        col_name = [str(i) for i in range(col_count)]

                    if(col_type is None):
                        col_type = int

                    if(not isinstance(col_type, list)):
                        col_type = [col_type for col  in col_name]
                    col_transform = lambda seq_list: [t(x) for x, t in zip(seq_list, col_type)]
                    dataset_type = cls.type_str[col_type[0]]
                if (head is not None and nb_lines + len(lines) > head):
                    if head - nb_lines == 0:
                        break
                    lines = lines[: head - nb_lines]
                nb_lines += len(lines)
                sys.stdout.flush()

                sys.stdout.flush()
                if(group_by_key is not None):
                    for line in lines:
                        line_list = line.split(split, len(col_name))
                        key = group_by_func(col_key_selector(line_list))
                        if key not in key_count:
                            key_count[key] = 0
                        key_count[key] += 1
                print('Nb lines processed: ' + str(nb_lines), end='\r')
                if(head is not None and nb_lines >= head):
                    break
        print("\n")
        if(group_by_key is None):
            key_count["data"] = nb_lines
        for i, (k, v) in zip(tqdm.trange(len(key_count)), key_count.items()):
            data_group.create_dataset(str(k), (v, len(col_name)) , dataset_type, compression=dataset_compression)
        count_lines = 0
        batch_dict = {}
        key_incremental_count = {}
        nb_iteration = nb_lines // batch_size  + (0 if(nb_lines % batch_size == 0) else 1)
        with open(csv_filepath, 'r') as csv_file:
            for i, lines in zip(tqdm.trange(nb_iteration), iter(lambda: tuple(islice(csv_file, batch_size)), ())):
                if(head is not None and count_lines + len(lines) > head):
                    lines = lines[: head - count_lines]
            
                count_lines += len(lines)
                for line in lines:
                    line_list = line.split(split, len(col_name))
                    key = group_by_func(col_key_selector(line_list))

                    if(group_by_key is None):
                        key = 'data'

                    if(key not in batch_dict):
                        batch_dict[key] = []

                    batch_dict[key].append(col_transform(line_list))

                for k, v in batch_dict.items():
                    if(k not in key_incremental_count):
                        key_incremental_count[k] = 0
                    data_group[str(k)][key_incremental_count[k]: key_incremental_count[k] + len(v)] = np.array(v)
                    key_incremental_count[k] += len(v)
                batch_dict = {}
