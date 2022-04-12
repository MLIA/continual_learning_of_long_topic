import os
import pandas as pd
import numpy as np
import json

def bm25_rank(queries_collection, documents_collection, storage_filepath, doc_storage_filepath=None, top_k=1000,
            threads=1):
    if not os.path.exists(os.path.join(doc_storage_filepath, 'index')):
        # compute the index
        print("Compute the index...")
        documents =\
            [{"id": str(doc_id), "contents": doc_content} for doc_id, doc_content in documents_collection]
        os.makedirs(os.path.join(doc_storage_filepath,'documents_collection'), exist_ok=True)
        with open(os.path.join(doc_storage_filepath,'documents_collection/documents.json'), 'w') as tmp_file_index:
            json.dump(documents, tmp_file_index) 

        command_line =\
            'python -m pyserini.index -collection JsonCollection \
            -generator DefaultLuceneDocumentGenerator -threads '+str(threads)+' -input '+\
            os.path.join(doc_storage_filepath, 'documents_collection') + ' -index ' +\
            os.path.join(doc_storage_filepath, 'index') + ' -storePositions -storeDocvectors -storeRaw'
        os.system(command_line)  

    if not os.path.exists(os.path.join(storage_filepath, 'bm25-' + str(top_k) + '.txt')):
        print("Creating queries tsv file")
        queries_df = pd.DataFrame(queries_collection)
        queries_df.to_csv(os.path.join(storage_filepath, 'queries.tsv') , sep='\t', header=False, index=False)
        print('Start retrieval')
        command_line = 'python -m pyserini.search --topics ' + os.path.join(storage_filepath, 'queries.tsv') + ' \
            --index ' + os.path.join(doc_storage_filepath, 'index') + ' \
            --output ' + os.path.join(storage_filepath, 'bm25-' + str(top_k) + '.txt') + ' \
            --bm25 --hits '+str(top_k)
        print(command_line)
        os.system(command_line)  
    print("loading scoreddocs")
    scoreddocs = pd.read_csv(os.path.join(storage_filepath, 'bm25-' + str(top_k) + '.txt'), sep=' ',
                                            header=None, usecols=[0, 2, 4],  names=['query_id', 'doc_id', 'score'],
                                            dtype={'query_id': str, 'doc_id':str, 'score': np.float32})
    return scoreddocs