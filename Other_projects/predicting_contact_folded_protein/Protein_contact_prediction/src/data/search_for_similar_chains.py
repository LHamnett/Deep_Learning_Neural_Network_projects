import yaml
import pandas as pd
import os
import numpy as np
import faiss
import pickle
from sklearn.neighbors import NearestNeighbors

def find_similar_with_length_filter(knn, 
                                    cls_matrix, 
                                    all_lengths, 
                                    query_cls, 
                                    Lq,
                                    k=5, 
                                    k_candidates=20, 
                                    len_filter_percent=0.2,
                                    train_query = True):
    """
    knn: fitted NearestNeighbors on CLS vectors
    cls_matrix: (N, d) matrix of CLS embeddings (only needed if knn doesn't store indices)
    all_lengths: np.ndarray of shape (N,) with training sequence lengths
    query_cls: np.ndarray of shape (d,)
    Lq: length of query sequence
    k: number of neighbours to return
    k_candidates: how many raw neighbours to ask KNN for
    min_ratio, max_ratio: allowed length range relative to query
    """
    query_cls = np.asarray(query_cls, dtype="float32").reshape(1, -1)

    # 1) get KNN candidates by cosine distance
    distances, idxs = knn.kneighbors(query_cls, n_neighbors=k_candidates)
    idxs = idxs[0]         # shape (k_candidates,)
    distances = distances[0]

    # 2) simple length filter
    Ls = all_lengths[idxs]
    lower = Lq * (1-len_filter_percent)
    upper = Lq * (1+len_filter_percent)
    keep_mask = (Ls >= lower) & (Ls <= upper)

    filtered_idxs = idxs[keep_mask]
    filtered_dists = distances[keep_mask]

    print(f'len filtered candidates:  {len(filtered_idxs)}')

    # 3) take first k that pass the filter
    if len(filtered_idxs) >= k:
        if train_query == True: #if comparing sequence that exists in training data, first idx   will be exact match
            return filtered_idxs[1:k+1]
        else:
            return filtered_idxs[:k]
    else:
        # not enough within range → just return what we have
        return filtered_idxs

def create_new_cols_for_similar_chains(df):

    # df['similar_chain_indices'] = df['closest_indices'].apply(lambda x: [i for i in x])
    df['similar_chain_sequences'] = df.apply(lambda row: [df.iloc[i]['residue_sequence'] for i in row['closest_indices']], axis=1)
    df['similar_contact_preds'] = df.apply(lambda row: [df.iloc[i]['predicted_contacts'] for i in row['closest_indices']], axis=1)
    
    #convert to separate columns:
    for i in range(df.iloc[0]['closest_indices'].__len__()):
        df[f'similar_chain_{i+1}_index'] = df['closest_indices'].apply(lambda x: x[i] if len(x) > i else None)
        df[f'similar_chain_{i+1}_sequence'] = df.apply(lambda row: row['similar_chain_sequences'][i] if len(row['similar_chain_sequences']) > i else None, axis=1)
        df[f'similar_chain_{i+1}_contact_pred'] = df.apply(lambda row: row['similar_contact_preds'][i] if len(row['similar_contact_preds']) > i else None, axis=1)
    return df


if __name__ == '__main__':

    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')

    with open(config_path,'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    merged_df_train_save_path = os.path.join(config['top_level_folder'],'data/ready_for_training/train_gt_embed_preds.pkl')
    merged_df = pd.read_pickle(merged_df_train_save_path)
    merged_df['seq_length'] = merged_df['residue_sequence'].apply(len)

    print(merged_df.columns)

    # merged_df.head()

    # rows_to_use = 2000
    rows_to_use = len(merged_df)
    train_knn_df = merged_df.iloc[:rows_to_use].reset_index(drop=True)
    
    emb_list   = train_knn_df["esm2_embeddings"].to_list()        # each: (d,)
    emb_matrix = np.stack([np.asarray(e, dtype="float32").reshape(-1) for e in emb_list], axis=0) #(N,d)
    all_lengths = train_knn_df["seq_length"].to_numpy()         # (N,)

    
    knn_fitted = NearestNeighbors(metric="cosine", algorithm="brute")
    knn_fitted.fit(emb_matrix)

    #testing single row
    # test_indices = find_similar_with_length_filter(
    #     knn=knn_fitted,
    #     cls_matrix = emb_matrix,
    #     all_lengths= all_lengths,
    #     query_cls=emb_matrix[0],
    #     Lq=all_lengths[0],
    #     k=5, k_candidates=100,
    #     len_filter_percent=0.1,
    #     train_query=True
    # )

    # print(test_indices)


    print('finding closest chain indices for training data')
    
    if 'closest_indices' not in train_knn_df.columns:

        train_knn_df['closest_indices'] = train_knn_df.apply(
                lambda row: find_similar_with_length_filter(
                    knn=knn_fitted,
                    cls_matrix=emb_matrix,
                    all_lengths=all_lengths,
                    query_cls=row['esm2_embeddings'],
                    Lq=row['seq_length'],       
                    k=5, k_candidates=100,
                    len_filter_percent=0.1,
                    train_query=True
                ),
        axis=1)

        train_knn_df.to_pickle(merged_df_train_save_path)
        print('added indices to train data file')
    
    else:
        print('closest_indices column already exists in dataframe, skipping computation')



    




