import os
import pandas as pd
import ast
from tqdm import tqdm
import argparse
import logging
from multiprocessing import Pool
import torch
from glob import glob

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(input_dir):
    logging.info(f"Loading batch files from directory: {input_dir}")

    batch_files = sorted(glob(os.path.join(input_dir, "batch*.tsv")))

    if not batch_files:
        logging.warning(f"No batch files found in directory: {input_dir}")
        return pd.DataFrame() 

    combined_df = pd.concat((pd.read_csv(file, sep='\t') for file in batch_files), ignore_index=True)
    logging.info(f"Loaded dataset shape: {combined_df.shape}")

    return combined_df

def process_embedding_column(df, emb_column, gpu_id):
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Processing {emb_column} on GPU {gpu_id}")

    emb_list = [ast.literal_eval(x) for x in tqdm(df[emb_column], desc=f"Processing {emb_column}")]
    emb_df = pd.DataFrame(emb_list)
    emb_df.columns = [f'feature_{i+1}' for i in range(emb_df.shape[1])]
    return emb_df

def process_batch(batch_file, emb_column, output_dir, gpu_id):
    logging.info(f"Processing batch: {batch_file} on GPU {gpu_id}")
    df = pd.read_csv(batch_file, sep='\t')
    df_filtered = df[df[emb_column].notna()].copy()
    emb_df = process_embedding_column(df_filtered, emb_column, gpu_id)

    emb_df['AUC'] = df['AUC']
    emb_df['label'] = df['label']
    emb_df['cancer_type'] = df['cancer_type']
    emb_df['cell_line_name'] = df['cell_line_name']
    emb_df['drug_name'] = df['drug_name']
    emb_df['Tissue'] = df['Tissue']
    emb_df['Tissue_sub_type'] = df['Tissue_sub_type']
    
    batch_filename = "feature_" + os.path.basename(batch_file).replace('.tsv', f'_{emb_column}.csv')
    output_path = os.path.join(output_dir, batch_filename)

    emb_df.to_csv(output_path, index=False)
    logging.info(f"Saved processed batch to {output_path} on GPU {gpu_id}")

    return output_path  

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description='Process embeddings and save to CSV files.')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the input directory containing batch files.')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save processed batch files.')
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='Comma-separated list of GPU IDs to use.')
    parser.add_argument('--dataset', type=str, default='CCLE_GDSCv2', help='Dataset name')
    parser.add_argument('--embedding_column', type=str, required=True, help='Embedding column to process')

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    batch_files = sorted(glob(os.path.join(args.input_dir, "batch*.tsv")))

    if not batch_files:
        logging.error(f"No batch files found in directory: {args.input_dir}")
        return

    gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(',')]
    num_gpus = len(gpu_ids)

    process_args = [(batch_file, args.embedding_column, args.output_dir, gpu_ids[i % num_gpus]) 
                    for i, batch_file in enumerate(batch_files)]

    with Pool(processes=min(num_gpus, len(batch_files))) as pool:
        processed_files = pool.starmap(process_batch, process_args)

    combined_df = pd.concat((pd.read_csv(file) for file in processed_files), ignore_index=True)

    final_output_path = os.path.join(args.output_dir, f"{args.dataset}_combined_{args.embedding_column}.csv")
    combined_df.to_csv(final_output_path, index=False)

    logging.info(f"Final combined dataset saved to {final_output_path}")

if __name__ == '__main__':
    main()
