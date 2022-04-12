import sys
import h5py

import os
import json
import pandas as pd

from lire.data_tools.dataset import MSMarco
from lire.data_tools import data_large

with open('/web/gerald/public_html/lire_data/msmarco_topics_cleaned_train_stsb-roberta-large.json', 'r') as infile:
    train_set = json.load(infile)
with open('/web/gerald/public_html/lire_data/msmarco_topics_cleaned_dev_stsb-roberta-large.json', 'r') as infile:
    dev_set   = json.load(infile)
with open('/web/gerald/public_html/lire_data/msmarco_topics_cleaned_eval_stsb-roberta-large.json', 'r') as infile:
    eval_set  = json.load(infile)

data_folder   = "/media/gerald/00B1B02B44A76AB2/CPD/data"
train_queries = MSMarco.MSMarcoPassageRankingDataset.load_queries("train", data_folder)
dev_queries   = MSMarco.MSMarcoPassageRankingDataset.load_queries("dev", data_folder)
eval_queries  = MSMarco.MSMarcoPassageRankingDataset.load_queries("eval", data_folder)
queries_set   = pd.concat([train_queries, dev_queries, eval_queries])

queries_set_unique = set(queries_set.index.unique().tolist())
data_folder = os.path.join(data_folder, "MSMarcoPassageRankingDataset")
dev_top_1000_path = os.path.join(data_folder, "dev-top-1000.data")
train_top_1000_path = os.path.join(data_folder, "train-id-top-1000.data")
train_triplet_path = os.path.join(data_folder,'train_positive_negative.data')

dev_inverse_cluster = {qid:cid for cid, c in enumerate(dev_set) for qid in c}
train_inverse_cluster = {qid:cid for cid, c in enumerate(train_set) for qid in c}

hdf5_filepath = os.path.join(data_folder, "data.hdf5")
# if(os.path.exists(hdf5_filepath)):
#     os.remove(hdf5_filepath)

# with h5py.File(hdf5_filepath,'a') as root:
#     dev = root.create_group("dev")
#     dev_1000 = dev.create_group("dev-top-1000")
#     train = root.create_group("train")
#     train_1000 = train.create_group("train-top-1000")
#     train_triple = train.create_group("qid-pid-nid-cluster")

data_large.HDF5DatasetManager.batched_csv_to_hdf5(hdf5_filepath, '/train/qid-pid-nid-cluster',
                                                  train_triplet_path, col_name=["qid", "pid", "nid"],
                                                  group_by_key="qid", group_by_func=lambda x : str(train_inverse_cluster[int(x)]),
                                                  col_type=int, batch_size=int(1e7), head=int(40e7)
                                                 )

data_large.HDF5DatasetManager.batched_csv_to_hdf5(hdf5_filepath, "dev/dev-top-1000",
                                                  dev_top_1000_path, col_name=["qid", "did"],
                                                  group_by_key="qid", col_type=int, batch_size=int(1e7)
                                                 )
# print("Training data")
# data_large.HDF5DatasetManager.batched_csv_to_hdf5(hdf5_filepath, "train/train-top-1000",
#                                                   train_top_1000_path, col_name=["qid", "did"],
#                                                   group_by_key="qid", col_type=int, batch_size=int(1e7)
#                                                  )