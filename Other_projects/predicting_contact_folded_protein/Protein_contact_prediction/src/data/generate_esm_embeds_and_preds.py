import torch
import pandas as pd
import os
# import transformers
import esm
import yaml
import copy
import numpy as np

def make_length_buckets(items, max_tokens=1200):
    '''
     items: list[(name, (label, seq))]
     '''
    # items = sorted(items, key=lambda kv: len(kv[1][1]))
    batch, acc = [], 0
    for name, (lbl, seq) in items:
        L = len(seq) + 2  # BOS/EOS
        if batch and acc + L > max_tokens:
            yield batch
            batch, acc = [], 0
        batch.append((name, (lbl, seq)))
        acc += L
    if batch:
        yield batch

def run_esm2_embeddings_and_contacts(
    sequences,
    model,
    alphabet,
    contacts_truth=None,
    eval_fn=None,
    layer=None,
    device=None,
    max_tokens=1200,
    precision='fp16',
    output_desired='embeds_only',
    max_seq_len_gpu=700,
    max_seq_len_total=1000,
    output_folder='',
    return_seq_embed_or_cls_token='cls_token'
):  

    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    if layer is None:
        layer = getattr(model, "num_layers", None) or max(getattr(model, "repr_layers", [33]))

    batch_converter = alphabet.get_batch_converter()

    # --- Initialise placeholders for every input sequence ---
    # means alignment between input and embeddings in maintained even if we skip processing certain sequences
    
    embeddings = {name: None for name in sequences}
    contact_preds = {name: None for name in sequences}
    skipped = {}

    # --- Filter sequences above hard length cap ---
    filtered_items = []
    for name, pair in sequences.items():
        seq = pair[1]
        if len(seq) > max_seq_len_total:
            skipped[name] = f"too_long ({len(seq)} > {max_seq_len_total})"
        else:
            filtered_items.append((name, pair))

    filtered_items = sorted(filtered_items, key=lambda kv: len(kv[1][1]))

    # --- Precision setup ---
    amp_dtype = None
    if device == "cuda" and precision:
        amp_dtype = torch.bfloat16 if precision.lower() == "bf16" else torch.float16

    cpu_model = None  # lazy CPU fallback

    for batch_num, batch_items in enumerate(make_length_buckets(filtered_items, max_tokens=max_tokens)):
        longest = max(len(seq) for _, (_, seq) in batch_items)
        use_cpu_for_batch = (device == "cuda" and longest > max_seq_len_gpu)

        # choose model/device
        if use_cpu_for_batch:
            if cpu_model is None:
                import copy
                cpu_model = copy.deepcopy(model).to("cpu").eval()
            active_model = cpu_model
            batch_device = "cpu"
        else:
            active_model = model
            batch_device = device

        data = [(name, seq) for name, (_, seq) in batch_items]
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(batch_device, non_blocking=True)

        print(f"[Batch {batch_num}] N={len(batch_items)} | max_len={longest} | device={batch_device}")

        try:
            # choose autocast context
            if batch_device == "cuda" and amp_dtype is not None:
                autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
            else:
                autocast_ctx = torch.autocast(device_type="cpu", enabled=False)

            with torch.inference_mode(), autocast_ctx:
                out = active_model(batch_tokens, repr_layers=[layer], need_head_weights=False)
                reps = out["representations"][layer]

                if output_desired in ("preds_only", "embeds_and_preds"): #don't get predict if not needed, memory intensive
                    pred_list = active_model.predict_contacts(batch_tokens)

            for batch_index, (seq_name, (_label, sequence)) in enumerate(batch_items):
                # Each sequence has length L (excluding BOS/EOS tokens)
                seq_length = len(sequence)

                # Extract the [L, d] embedding for this sequence
                if return_seq_embed_or_cls_token == 'seq_embed':
                # model outputs include BOS and EOS tokens, so slice [1 : 1+L]
                    embedding_out = reps[batch_index, 1:1 + seq_length, :].float().cpu()
                elif return_seq_embed_or_cls_token == 'cls_token':
                    embedding_out = reps[batch_index, 0, :].float().cpu().unsqueeze(0)  # [1, d]
                
                # Save embedding
                embeddings[seq_name] = embedding_out

                if output_desired in ("preds_only", "embeds_and_preds"):
                    # Each prediction is an [L, L] contact probability matrix
                    seq_contact_map = pred_list[batch_index].float().cpu()
                    # contact_preds[seq_name] = seq_contact_map

                    
                    if output_desired == 'embeds_and_preds':
                        file_suffix = 'embeds_preds'
                        output_path = os.path.join(output_folder,f"{seq_name}_{file_suffix}.npz")
                        np.savez(output_path, embeddings=embedding_out, contact_map=seq_contact_map)

                    elif output_desired == 'preds_only':
                        file_suffix = 'contact_preds'
                        output_path = os.path.join(output_folder,f"{seq_name}_{file_suffix}.npy")
                        np.savez(output_path,seq_contact_map)
                
                else: #just save embeddings
                    file_suffix = 'embeds'
                    output_path = os.path.join(output_folder,f"{seq_name}_{file_suffix}.npy")
                    np.savez(output_path,seq_embedding.numpy())

        except RuntimeError as e:
            err_msg = str(e).split("\n")[0]
            print(f"⚠️ Batch {batch_num} failed ({err_msg})")
            for name, _ in batch_items:
                skipped[name] = f"runtime_error: {err_msg}"
            continue

        finally:
            #tidy up after each batch to prevent OOM errors
            del batch_tokens
            if 'reps' in locals(): del reps
            if 'pred_list' in locals(): del pred_list
            if 'out' in locals(): del out
            if device == "cuda":
                torch.cuda.empty_cache()

            

    print(f"✅ Completed.")

    

    # return embeddings, contact_preds if any(v is not None for v in contact_preds.values()) else None, skipped

    return


if __name__ == '__main__':

    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')

    with open(config_path,'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    train_data_pkl_path = config['train_data_combined_filepath']

    training_df = pd.read_pickle(train_data_pkl_path)

    embeds_preds_folder_path = os.path.join(config['top_level_folder'],'data/embeds_preds')

    embeds_train_folder_path = os.path.join(embeds_preds_folder_path,'train')

    os.system('rm -rf ' + embeds_train_folder_path)

    if not os.path.exists(embeds_train_folder_path):
        os.makedirs(embeds_train_folder_path)

    # train_df[]

    if 'esm2_embeddings' not in training_df.columns:

        training_df['uniq_id'] = training_df['filename'].apply(lambda x: x.strip('.npy')) + '_ch_' + training_df['chain_number'].astype(str)
        
        # num_elems_to_use = 64
        num_elems_to_use = len(training_df)
        all_uniq_ids = training_df['uniq_id'].values[0:num_elems_to_use]
        all_residue_sequences = training_df['residue_sequence'].values[0:num_elems_to_use]
        all_contact_maps = training_df['contact_map'].values[0:num_elems_to_use]

        #make sequence dict
        sequence_dict = {}
        for i, (filename, sequence) in enumerate(zip(all_uniq_ids, all_residue_sequences)):
            sequence_dict[filename] = (filename, sequence)

            '''
        available models hugging face
        esm2_t48_15B_UR50D	48	15B
        esm2_t36_3B_UR50D	36	3B
        esm2_t33_650M_UR50D	33	650M
        esm2_t30_150M_UR50D	30	150M
        esm2_t12_35M_UR50D	12	35M
        esm2_t6_8M_UR50D	6	8M

        '''
        #init models and esm  batch functions
        # esm2_model, esm2_alphabet =esm.pretrained.esm2_t33_650M_UR50D()
        # esm2_model, esm2_alphabet =esm.pretrained.esm2_t30_150M_UR50D()
        esm2_model, esm2_alphabet =esm.pretrained.esm2_t12_35M_UR50D()
        
        esm2_batch_converter = esm2_alphabet.get_batch_converter()

        #
        # embeddings, contact_preds,_   = run_esm2_embeddings_and_contacts(
        run_esm2_embeddings_and_contacts(
            sequences=sequence_dict,
            model=esm2_model,                          # loaded ESM2 model
            alphabet=esm2_alphabet,                       # its alphabet
            contacts_truth=None,            # optional dict: name -> (L,L) ground-truth contact map
            eval_fn=None,                   # optional callable(pred_map, true_map) -> dict of metrics
            layer=None,                     # which representation layer to extract (defaults to last)
            # device='cpu',                    # "cuda" or "cpu"; auto if None
            # batch_size=4,                    # small batch size to be safe on CPU/GPU RAM
            precision='fp16',
            # output_desired = 'embeds_only'
            output_desired='embeds_and_preds',
            output_folder = embeds_train_folder_path
        )

        # Files are now saved directly, so we can't access embeddings and contact_preds
        print("Embeddings and predictions saved to files successfully.")

        # training_df['esm2_embeddings'] = training_df['filename'].map(embeddings)
    # # training_df['contact_preds'] = training_df['filename'].map(contact_preds)

    # # training_df.head()

