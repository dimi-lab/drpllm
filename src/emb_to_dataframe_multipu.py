import os
import pandas as pd
import ast
from tqdm import tqdm
import argparse
import logging
from multiprocessing import Pool, cpu_count
import torch

def setup_logging():
    """
    Set up logging for the script.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(input_file):
    logging.info(f"Loading input file: {input_file}")
    combined_df = pd.read_csv(input_file, sep='\t')
    logging.info(f"Loaded dataset shape: {combined_df.shape}")
    
    return combined_df

def process_embedding_column(df, emb_column, gpu_id):
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Processing {emb_column} on GPU {gpu_id}")

    emb_list = [ast.literal_eval(x) for x in tqdm(df[emb_column], desc=f"Processing {emb_column}")]
    emb_df = pd.DataFrame(emb_list)
    emb_df.columns = [f'feature_{i+1}' for i in range(emb_df.shape[1])]
    return emb_df

def process_and_save_embedded_df(embedding_info):
    df, emb_column, output_filename, gpu_id = embedding_info
    emb_df = process_embedding_column(df, emb_column, gpu_id)
    emb_df['AUC'] = df['auc']
    emb_df['label'] = df['auc_disc']
    emb_df['cancer_type'] = df['cancer_type']

    emb_df.to_csv(output_filename, index=False)
    logging.info(f"Saved {emb_column} to {output_filename} on GPU {gpu_id}")

def main():
    # Set up logging
    setup_logging()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process embeddings and save to CSV files.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input file.')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save output CSV files.')
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='Comma-separated list of GPU IDs to use.')
    parser.add_argument('--dataset', type=str, default='CCLE', help='dataset name')
    args = parser.parse_args()

    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load the dataset
    combined_df = load_data(args.input_file)

    # Define embedding columns and corresponding output filenames
    embeddings_info = [
        (combined_df, 'emb_question_prompt', os.path.join(args.output_dir, args.dataset + '_question_emb_data.csv')),
        (combined_df, 'emb_question_prompt_0', os.path.join(args.output_dir, args.dataset + '_question0_emb_data.csv')),
        (combined_df, 'emb_question_prompt_1', os.path.join(args.output_dir, args.dataset + '_question1_emb_data.csv')),
        (combined_df, 'emb_question_prompt_2', os.path.join(args.output_dir, args.dataset + '_question2_emb_data.csv')),
        (combined_df, 'emb_refined_prompt_few_shot', os.path.join(args.output_dir, args.dataset + '_few_shot_emb_data.csv')),
        (combined_df, 'emb_refined_prompt_context', os.path.join(args.output_dir, args.dataset + '_context_emb_data.csv')),
        (combined_df, 'emb_refined_prompt_cell', os.path.join(args.output_dir, args.dataset + '_cellline_context_emb_data.csv')),
        (combined_df, 'emb_refined_prompt_drug', os.path.join(args.output_dir, args.dataset + '_drug_context_emb_data.csv'))
    ]

    # Assign GPUs in a round-robin fashion
    gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(',')]
    num_gpus = len(gpu_ids)

    # Assign GPU ID to each task
    embeddings_info = [(emb_info[0], emb_info[1], emb_info[2], gpu_ids[i % num_gpus]) for i, emb_info in enumerate(embeddings_info)]

    # Set up multiprocessing
    num_processes = min(num_gpus, len(embeddings_info))  # Ensure we don't start more processes than GPUs
    with Pool(processes=num_processes) as pool:
        pool.map(process_and_save_embedded_df, embeddings_info)

if __name__ == '__main__':
    main()
