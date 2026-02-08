import os
import pandas as pd
from biopandas.pdb import PandasPdb
import yaml
from typing import List
import numpy as np
from scipy.spatial.distance import pdist, squareform
import pickle
from typing import Tuple
import timeit
from concurrent.futures import ProcessPoolExecutor, as_completed
import uuid
from pathlib import Path

#help functions
# mapping dict
RES3_TO_RES1 = {
    "ALA": "A",
    "ARG": "R",
    "ASP": "D",
    "CYS": "C",
    "CYX": "C",  
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "HIE": "H",  
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "ASN": "N",
    "PHE": "F",
    "PRO": "P",
    "SEC": "U",  
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

def residue_to_one_letter(residue_name: str) -> str:
    """
    Convert a 3-letter residue name to its 1-letter amino acid code.
    Returns 'X' for unknown or non-standard residues.
    """
    residue_name = residue_name.strip().upper()
    return RES3_TO_RES1.get(residue_name, "X")


#processing input data
def extract_ca_single_pdb_file(file_path:str) -> PandasPdb:

    #check if more than one model - how to handle?

    ppdb = PandasPdb().read_pdb(file_path)
    
    if 'MODEL' in ppdb.df.keys():
        print(f"Warning: Multiple models found in {file_path}. Only the first model will be processed.")
        ppdb.df['ATOM'] = ppdb.df['ATOM'][ppdb.df['ATOM']['model_num'] == 1]

    df = ppdb.df['ATOM']
    cols_to_drop = [col for col in df.columns if 'blank' in col]
    df = df.drop(columns=cols_to_drop)

    #only interested in CA entries
    ca_df = df[df["atom_name"] == "CA"]

    return ca_df

def convert_ca_df_into_per_chain_data(ca_df: PandasPdb) -> List[PandasPdb]:

    unique_chains = ca_df.chain_id.unique()
    dfs_per_chain = []
    for chain in unique_chains:
        temp_df = ca_df.copy()
        temp_df = temp_df[temp_df['chain_id'] == chain]
        dfs_per_chain.append(temp_df)

    return dfs_per_chain

def make_residue_seq_per_chain(chain_df: PandasPdb) -> str:

    chain_df['residue_single_letter'] = chain_df['residue_name'].map(residue_to_one_letter)
    letter_list = chain_df['residue_single_letter'].to_list()
    return ''.join(letter_list)



    


#making binary contact target per chain

def convert_coords_to_distance_matrix_single_chain(single_chain_df: PandasPdb) -> np.array:

    all_coords = np.asarray(single_chain_df[['x_coord','y_coord','z_coord']])

    return squareform(pdist(all_coords, metric='euclidean')).astype(np.float32, copy=False)

def convert_dist_matrix_to_binary_contact_map(dist_matrix,contract_threshhold=8):

    binary_contact_map = (dist_matrix < contract_threshhold).astype(np.uint8)

    np.fill_diagonal(binary_contact_map, 0)

    return binary_contact_map

## do preprocessing
def apply_preprocessing_single_file(input_filepath):

    ca_df = extract_ca_single_pdb_file(input_filepath)
    dfs_per_chain = convert_ca_df_into_per_chain_data(ca_df)
    processed_data_pairs = []
    for chain in dfs_per_chain:
        single_letter_res_seq = make_residue_seq_per_chain(chain)
        dist_map = convert_coords_to_distance_matrix_single_chain(chain)
        bin_contact_map = convert_dist_matrix_to_binary_contact_map(dist_map, contract_threshhold=8)
        processed_data_pairs.append([single_letter_res_seq,bin_contact_map])

    return processed_data_pairs


def _process_one_file_to_single_npy(src_path: str, out_dir: str) -> Tuple[str, int]:
    """
    Process a single PDB file and write exactly one .npy file containing
    a list of (chain_idx, input_seq, bin_mask_uint8) tuples.

    Returns: (output_npy_path, num_chains_written)
    """
    pairs = apply_preprocessing_single_file(src_path)  # [[seq, mask], ...]
    records = []
    for chain_idx, (seq, mask) in enumerate(pairs):
        # store compactly
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8, copy=False)
        records.append((int(chain_idx), str(seq), mask))

    # Ensure output dir exists
    os.makedirs(out_dir, exist_ok=True)

    # Name: stem + random suffix to avoid collisions during parallel writes
    # uid = uuid.uuid4()
    out_name = f"{Path(src_path).stem}.npy"
    out_path = os.path.join(out_dir, out_name)

    
    
    arr = np.array(records, dtype=object)
    # Write to the correct output directory
    np.save(out_path, arr, allow_pickle=True)

    return out_path, len(records)

def process_files_parallel_to_npy(
    files_to_process: List[str],
    output_dir: str,
    max_workers: int | None = None,
) -> None:
    """
    CPU-bound parallel processing. For each input file, writes ONE .npy file
    containing [(chain_idx, input_seq, bin_mask_uint8), ...].
    Prints progress; does not accumulate large data in RAM.
    """
    os.makedirs(output_dir, exist_ok=True)
    total = len(files_to_process)
    done = 0

    with ProcessPoolExecutor(max_workers=(max_workers or os.cpu_count() or 1)) as ex:
        futs = {
            ex.submit(_process_one_file_to_single_npy, fp, output_dir): fp
            for fp in files_to_process
        }
        for fut in as_completed(futs):
            src = futs[fut]
            try:
                out_path, n = fut.result()
                done += 1
                print(f"\r[{done}/{total}] {src} -> {out_path} ({n} chains)", end='', flush=True)
            except Exception as e:
                done += 1
                print(f"\r[{done}/{total}] ERROR {Path(src).name}: {e}", end='', flush=True)
    
    print("\nDone.")
    return done

if __name__ == "__main__":
    
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')

    with open(config_path,'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    raw_data_folder = os.path.join(config['top_level_folder'],'data/raw')
    config['raw_data_folder'] = raw_data_folder

    raw_train_data_folder = os.path.join(raw_data_folder,'train')
    config['raw_train_data_folder'] = raw_train_data_folder

    train_filepaths = [os.path.join(raw_train_data_folder,file) for file in os.listdir(raw_train_data_folder)]

    # train_files_to_process = train_filepaths[0:200]
    train_files_to_process = train_filepaths
    # processed_data = apply_preprocessing_single_file(train_file)

    # print(processed_data[0])

    processed_data_folder = os.path.join(config['top_level_folder'],'data/processed')
    config['processed_data_folder'] = processed_data_folder

    output_path = os.path.join(processed_data_folder,'train_data_processed')
    config['train_data_processed_folder'] = output_path

    #save new config variables
    with open(config_path, 'w') as yaml_file:
        yaml.dump(config, yaml_file)

    # max_workers = os.cpu_count()
    
    # start_time = timeit.default_timer()
    # files_processed = process_files_parallel_to_npy(train_files_to_process,output_dir=output_path,max_workers=max_workers)
    # end_time = timeit.default_timer()
    
    # print(f"Processed {files_processed}, completed in {end_time - start_time:.2f} seconds.")

    

    

