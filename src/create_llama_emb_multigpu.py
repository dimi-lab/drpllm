import os
import pandas as pd
import numpy as np
import json
import gc
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import nn
from multiprocessing import Pool, Queue, current_process
from datetime import datetime

class LlamaSentenceEmbedding:
    def __init__(self, model_path, device='cuda:0', max_length=512, output_size=3072):
        torch.cuda.empty_cache() 
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = torch.device(device)
        self.max_length = max_length

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16,
        ).to(self.device)

        self.projection_layer = nn.Linear(self.model.config.hidden_size, output_size).to(self.device).to(torch.float16)

    def get_hidden_state_before_response(self, texts: list[str]):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                                max_length=self.max_length).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
        
        final_hidden_states = hidden_states[:, -1, :].to(torch.float16)
        projected_hidden_states = self.projection_layer(final_hidden_states)
        return projected_hidden_states.to(torch.float16)

    def encode(self, sentences: list[str]) -> list[np.ndarray]:
        if len(sentences) == 0:
            return []
        embeddings = self.get_hidden_state_before_response(sentences)
        return [embedding for embedding in embeddings]

def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()
    

def generate_response_and_embed_batch_llama(text_list, llama_embedder, batch_size=8):
    try:
        batched_embeddings = []
        for i in tqdm(range(0, len(text_list), batch_size), desc="Processing Batches"):
            batch = text_list[i:i + batch_size]
            with torch.no_grad():
                batch_embeddings = llama_embedder.encode(batch)
            batched_embeddings.extend(batch_embeddings)
            clear_memory()
        return batched_embeddings
    except Exception as e:
        print(f"Error processing batch: {e}")
        return [None] * len(text_list)


def process_chunk_llama(chunk, model_path, batch_size, max_length, gpu_id, output_folder):
    device = f'cuda:{gpu_id}'

    additional_columns = ['AUC', 'label', 'cancer_type', 'cell_line_name', 'drug_name']
    optional_columns = ['Tissue', 'Tissue_sub_type']
#    additional_columns.extend([col for col in optional_columns if col in data_df.columns])

    llama_embedder = LlamaSentenceEmbedding(model_path=model_path, device=device, max_length=max_length)

    columns_to_embed = ['refined_prompt_cell',
                        'refined_prompt_context',
                        'refined_prompt_drug']



    for column in tqdm(columns_to_embed, desc=f"Processing Columns on GPU {gpu_id}", leave=True):
        sentences = chunk[column].fillna("").tolist()
        embeddings = generate_response_and_embed_batch_llama(sentences, llama_embedder, batch_size=batch_size)
        chunk.loc[:, f'emb_{column}'] = [json.dumps(emb.tolist()) if emb is not None else "" for emb in embeddings]
        tqdm.write(f"Processed column: {column} on GPU {gpu_id}")

    return chunk

def load_data(data_path, sample_size=100000, random_state=42):
    df = pd.read_csv(data_path, sep='\t')
    

    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state)
        return df
    else:
        return df



def save_data(processed_chunk, output_folder):
    output_file = output_folder + "/" + 'batch_emb_prompt_llama_8b.tsv'
    processed_chunk.to_csv(output_file, sep='\t', index=False)


def init_worker(gpu_ids):
    process_id = current_process()._identity[0] - 1  # _identity[0] gives worker index (1-based)
    gpu_id = gpu_ids[process_id % len(gpu_ids)]  # Distribute GPU IDs cyclically
    return gpu_id


def main():
    parser = argparse.ArgumentParser(description='Run LLaMA embeddings for a given dataset on multiple GPUs')
    parser.add_argument('--input', type=str, required=True, help='Path to input TSV file')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save output batch files')
    parser.add_argument('--model_path', type=str, default='meta-llama/Meta-Llama-3.1-8B', help='Hugging Face model path')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for embedding')
    parser.add_argument('--max_length', type=int, default=9200, help='Maximum sequence length')
    parser.add_argument('--device_ids', type=int, nargs='+', default=[0, 1, 2, 3], help='List of GPU IDs to use')
    parser.add_argument('--chunk_size', type=int, default=5000, help='Number of rows per chunk')

    args = parser.parse_args()

    data = load_data(args.input)

    chunks = [data[i:i + args.chunk_size] for i in range(0, len(data), args.chunk_size)]

    with Pool(processes=len(args.device_ids)) as pool:
        results = [pool.apply_async(process_chunk_llama, (chunk, args.model_path,
                                                          args.batch_size, args.max_length, gpu_id, args.output_folder))
           for chunk, gpu_id in zip(chunks, args.device_ids * (len(chunks) // len(args.device_ids) + 1))]

        processed_chunks = [res.get() for res in results]

    processed_data = pd.concat(processed_chunks, ignore_index=True)
    save_data(processed_data, args.output_folder)

    print(f"Processing completed. Data saved to {args.output_folder}")


if __name__ == "__main__":
    main()
