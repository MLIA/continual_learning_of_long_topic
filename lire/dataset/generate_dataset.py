''' Clustering query to build tasks.
The following script create a train/val/test set with(n, 20, 40) queries respectivelly
'''

import pandas as pd
import numpy as np
import argparse
import pickle
import os
import torch
import tqdm
import random
import inspect
from sentence_transformers import SentenceTransformer, util



class ContinualGenerator():
    ''' Continual generator that split corpus based on topics.

        Parameters:

            query_set : [(query_id, query_txt)]
                The set of queries information (query_id and the text 
                associated to the query)
            pre_computed_embedding_path : str(path) Optional
                The path to the queries embedding if not given
                embeddings will be computed instead (can take time).
                If no file is available at the current path but the 
                parameter is given the embedding will be saved at the 
                location

    '''
    def __init__(self, query_set, pre_computed_embedding_path=None):
        self.query_set = query_set
        self.pre_computed_embedding_path = pre_computed_embedding_path


    def _get_embeddings(self, queries_content): 
        if self.pre_computed_embedding_path is None:
            model = SentenceTransformer('stsb-roberta-large')
            corpus_embeddings = model.encode(queries_content, show_progress_bar=True, convert_to_numpy=True)

        elif not os.path.exists(self.pre_computed_embedding_path):
            os.makedirs(os.path.dirname(self.pre_computed_embedding_path), exist_ok=True)
            model = SentenceTransformer('stsb-roberta-large')
            corpus_embeddings = model.encode(queries_content, show_progress_bar=True, convert_to_numpy=True)

            with open(self.pre_computed_embedding_path, "wb") as fOut:
                pickle.dump({'sentences': queries_content , 'embeddings': corpus_embeddings}, fOut)

        else:
            with open(self.pre_computed_embedding_path, "rb") as fOut:
                corpus_embeddings = pickle.load(fOut)['embeddings']

        return corpus_embeddings
    
    @staticmethod
    def community_filtering(cos_scores, alpha):
        return torch.arange(len(cos_scores))[cos_scores >= alpha]

    @staticmethod
    def community_detection_clustering(embeddings, ss_embeddings_estimation=50000,
                                       alpha=0.75, beta=0.55, mcs=2000):
        rp = torch.randperm(len(embeddings))
        sse = rp[:ss_embeddings_estimation]
        ss_embeddings = embeddings[sse]
        emcs = round((ss_embeddings_estimation/len(embeddings)) * mcs)
        print('Computing the cos score on ', len(ss_embeddings), ' embeddings')
        cos_scores = util.pytorch_cos_sim(ss_embeddings, ss_embeddings)

        print('Retrieve the ', emcs, ' closest neigbhors for each embeddings (max similarity)')
        top_k_values, top_k_indexes = cos_scores.topk(k=emcs, largest=True)

        print('Create the communities')
        communities = [(rp[top_k_indexes[i][0]], rp[ContinualGenerator.community_filtering(cos_scores[i], alpha)]) 
                    for i in range(len(top_k_values))  
                    if(top_k_values[i][-1] >= alpha)]
        print('Filtering communities to avoid overlapping (at the time there is ', len(communities),' communities)')
        sorted_communities = sorted(communities, key=lambda x: len(x[1]), reverse=True)

        unique_communities = []
        extracted_ids = set()

        for centroid, community in sorted_communities:
            add_cluster = True
            for idx in community:
                if idx in extracted_ids:
                    add_cluster = False
                    break

            if add_cluster:
                unique_communities.append((centroid, community))
                for idx in community:
                    extracted_ids.add(idx)
        centroids = [torch.Tensor(embeddings[centroid]) for centroid, _ in unique_communities]

        print('creating real clusters according to number of examples by clusters')
        print(len(embeddings))
        cluster_center = torch.stack(centroids)
        complete_communities = [[] for i in range(len(cluster_center))]
        for i, d in enumerate(embeddings):
            cos_sim = util.pytorch_cos_sim(cluster_center, d).flatten()
            max_value, max_index = cos_sim.max(-1)
            if max_value >= beta:
                complete_communities[max_index].append(i)
        
    
        return centroids, complete_communities




    def generate(self, t1=0.75, t2=0.5, mcs=2000, estimation_set_size=50000):
        ''' Generate groups of queries.
            
            Parameters:
                t1: float
                    the first threshold
                t2: float
                    the second threshold (all dataset)
                msc: int
                    minimum number of groups representent
                estimation_set_size: int 
                    the number of elements to perform clustering
                    according to t1
        '''
        queries_ids = [k for (k, v) in self.query_set]
        queries_content = [v for (k, v) in self.query_set]
        embeddings = self._get_embeddings(queries_content)
        centroids, communities =\
            ContinualGenerator.community_detection_clustering(embeddings, estimation_set_size, t1, t2, mcs)
        fcom = [[queries_ids[query_index] for j, query_index in enumerate(community)]
                 for i, community in enumerate(communities)]
        return fcom, centroids

def create_new_continual_set(self, queries_set, documents_set,
                             queries_embeddings_path=None, documents_bm25_index_path=None,
                             t1=0.75, t2=0.5, mcs=2000, estimation_set_size=50000,
                             val_size=50, test_psize=50, topk=1000):
    def split_set(queries, val_set_size, test_set_size):
        queries = queries[torch.randperm(len(queries))]
        end_val = val_set_size if(isinstance(val_set_size, int)) else int(val_set_size * len(queries))
        end_test = end_val + test_set_size if(isinstance(test_set_size, int)) else val_set_size + int(test_set_size * len(queries))
        assert(end_val + end_test <= len(queries))
        return {'val': queries[:end_val], 'test': queries[end_val:end_test], 'train': queries[end_test:]}

    #TODO
    bm = ranker.BM25('/net/cal/gerald/CPD/data/set')
    bm.create_index(ms_documents, '/net/cal/gerald/CPD/data/set')
    topk_documents = bm.rank_corpus(queries_final, top_k=topk)



# print('Filtering according to existance of triplet')
# csv = train_dataset.load_triplets()
# query_set_check = set([query for cluster in clusters for query in cluster])
# a = csv.index.isin(query_set_check)
# b = set(np.unique(csv.index[a]).tolist())
# if(query_set_check != len(b)):
#     print("Missing some queries in triplet file train : "+ str(len(query_set_check - b)))
# print("Number of triplets : " + str(a.sum()) + " for " + str(len(b))+" queries")
# assert(len(a) == len(csv))
# triplet_subset = csv.iloc[a]
# query_available = query_available.intersection(set(triplet_subset.index))

# print("Process the clusters")
# clusters = [[query for query in cluster if(query in query_available)] for cluster in clusters]


# print(len(clusters), '  final clusters find with the following size ', [len(cluster) for cluster in clusters])


# example_cluster = [[queries_content[inv_queries_id[index]] for index in topic[:2]] for topic in clusters]

# print('Representative samples by clusters : ', example_cluster)


# print('Get the closest neigbhor documents according to bm25 (if you are not satisfy and want to change k of top k use the prediction/bm25_triplet script)')

# print('Loading documents')
# documents = train_dataset.load_documents()
# ms_documents = {i_doc: documents[i_doc][0] for i_doc in documents}

# print('Transform queries for correct format')

# queries_final  = {qid: queries_dct[qid][0] for qid in query_available}

# print('computing top100 for current documents')

# bm = ranker.BM25('/net/cal/gerald/CPD/data/set')
# bm.create_index(ms_documents, '/net/cal/gerald/CPD/data/set')
# topk_documents = bm.rank_corpus(queries_final, top_k = args.topk)

# print('Splitting into train val test')

# def split_set(queries, val_set_size, test_set_size):
#     queries = queries[torch.randperm(len(queries))]
#     end_val = val_set_size if(isinstance(val_set_size, int)) else int(val_set_size * len(queries))
#     end_test = end_val + test_set_size if(isinstance(test_set_size, int)) else val_set_size + int(test_set_size * len(queries))
#     assert(end_val + end_test <= len(queries))
#     return {'val': queries[:end_val], 'test': queries[end_val:end_test], 'train': queries[end_test:]}


# clusters = [split_set(torch.LongTensor(cluster), args.val_set_size, args.test_set_size) for cluster in clusters]

# print([{'val':len(s['val']), 'train': len(s['train']), 'test':len(s['test'])} for s in c  for c in clusters])
# print('Saving the dataset: it can take a while ')


