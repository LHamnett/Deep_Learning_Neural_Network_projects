import numpy as np
import os
import pandas as pd
import yaml



def make_training_df(folder_path):

    processed_training_files = os.listdir(folder_path)
    processed_training_filepaths = [os.path.join(folder_path,file) for file in processed_training_files]
    filenames = []
    chain_nums = []
    res_sequences = []
    contact_maps = []

    for i,path in enumerate(processed_training_filepaths):
        
        temp_data = np.load(path, allow_pickle=True)

        # print(temp_data)
        temp_filename = processed_training_files[i]

        filenames.append(temp_filename)
        chain_nums.append(temp_data[0][0]+1)
        res_sequences.append(temp_data[0][1])
        contact_maps.append(temp_data[0][2])

        
        

    print('finished reading embed and pred files')

    df = pd.DataFrame({
        'filename': filenames,
        'chain_number': chain_nums,
        'residue_sequence': res_sequences,
        'gt_contact_map': contact_maps
    })

    return df

def add_embeds_and_preds_to_df(input_df, embedding_folder_path):

    filenames = os.listdir(embedding_folder_path)
    filenames = [file for file in filenames if file.endswith('.npz')]
    embedding_file_paths = [os.path.join(embedding_folder_path,file) for file in filenames]
    
    uniq_ids = []
    embeddings = []
    contact_map_preds = []
    correctly_opened = 0
    error_opening = 0

    for i,filepath in  enumerate(embedding_file_paths):
        if i % 100 == 0:
            print(f"Loading embeds and pred file {i+1} of {len(embedding_file_paths)}")

        try:
            temp_array = np.load(filepath)
            temp_embedding = temp_array['embeddings']
            temp_contact_pred = temp_array['contact_map']
            uniq_ids.append(filenames[i].strip('_embeds_preds.npz')) 
            embeddings.append(temp_embedding)
            contact_map_preds.append(temp_contact_pred)
            correctly_opened += 1

        except Exception as e:
            print(f"Error loading file: {filepath} with {e} Skipping.")
            print(f'keys in file: {temp_array.files}')
            error_opening += 1
            continue

        # if i > 500:
        #     break

    print(f"Finished loading embeddings and contact map predictions. Successfully opened {correctly_opened} files, {error_opening} errors.")

    embed_contact_df = pd.DataFrame({
        'uniq_id': uniq_ids,
        'esm2_embeddings': embeddings,
        'esm2_contact_map_preds': contact_map_preds
    })

    return embed_contact_df




if __name__ == "__main__":
    
    #define vars
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')

    with open(config_path,'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    
    train_data_combined_folder = 'data/processed/train_data_combined'
    train_data_combined_path = os.path.join(config['top_level_folder'],train_data_combined_folder)
    config['train_data_combined_folder'] = train_data_combined_path

    train_data_combined_filepath = os.path.join(train_data_combined_path,'training_data_df.pkl')
    config['train_data_combined_filepath'] = train_data_combined_filepath

    regen_train_df = False
    
    #make df with input seqs, ground truth contact maps - reload if already made
    if regen_train_df == True:
    
        training_df = make_training_df(config['train_data_processed_folder'])

        # training_df.to_pickle(train_data_combined_filepath)
        print("Combined Training DataFrame saved to:", train_data_combined_path)
    
    else:
        training_df = pd.read_pickle(train_data_combined_filepath)

    training_df['uniq_id'] = training_df['filename'].str.replace('.npy','') + '_ch_' + training_df['chain_number'].astype(str)
    # print(training_df.columns)
    # print(training_df.head(5))

    #read esm2 embedding and pred files
    train_embed_preds_folder = '/home/leon/Documents/github_repos/deep_origin_take_home_assignment/deep_origin/data/embeds_preds/train'
    embed_contact_df = add_embeds_and_preds_to_df(training_df, train_embed_preds_folder)
    
    #create combined df
    merged_df = pd.merge(training_df, embed_contact_df, on='uniq_id', how='inner')
    print(merged_df.head(5))
    
    

    #remove rows where esm2 contact prediction shape doesn't match input_seq length or ground truth contact shape

    merged_df['gt_contact_map_shapes'] = merged_df['contact_map'].apply(lambda x: x.shape)
    merged_df['esm2_cont_preds_shapes'] = merged_df['esm2_contact_map_preds'].apply(lambda x: x.shape)

    mismatched_shape_df = merged_df[merged_df['gt_contact_map_shapes'] != merged_df['esm2_cont_preds_shapes']]

    print(f'number of train data rows with mismatched shape between contact pred and gt: {mismatched_shape_df.shape[0]}')

    merged_df_bad_rows_dropped = merged_df.drop(mismatched_shape_df.index)
    
    #save output 
    print('saving cleaned train data (takes time to save)')

    merged_df_train_save_path = os.path.join(config['top_level_folder'],'data/ready_for_training/train_gt_embed_preds.pkl')
    config['merged_cleaned_df_path'] = merged_df_train_save_path

    merged_df_bad_rows_dropped.to_pickle(merged_df_train_save_path)

    print('saved ground truth, embeddings, and predictions for train data')

    with open (config_path, 'w') as yaml_file:
        yaml.dump(config, yaml_file)
    


