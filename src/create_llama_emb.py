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

# Define the LlamaSentenceEmbedding class
class LlamaSentenceEmbedding:
    def __init__(self, model_path, device='cuda:0', max_length=512, output_size=3072):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = torch.device(device)
        self.max_length = max_length
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16
        ).to(self.device)

        self.projection_layer = nn.Linear(self.model.config.hidden_size, output_size).to(self.device).to(torch.float16)

    def get_hidden_state_before_response(self, texts: list[str]):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.device)
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

# Function to process batches
def generate_response_and_embed_batch_llama(text_list, llama_embedder, batch_size=8):
    try:
        batched_embeddings = []
        for i in tqdm(range(0, len(text_list), batch_size), desc="Processing Batches"):
            batch = text_list[i:i + batch_size]
            batch_embeddings = llama_embedder.encode(batch)
            batched_embeddings.extend(batch_embeddings)
            torch.cuda.empty_cache()
            gc.collect()
        return batched_embeddings
    except Exception as e:
        print(f"Error processing batch: {e}")
        return [None] * len(text_list)

# Function to process the chunk
def process_chunk_llama(chunk, llama_embedder, batch_size=8):
    columns_to_embed = ['refined_prompt_context', 'refined_prompt_drug','refined_prompt_few_shot']
    for column in tqdm(columns_to_embed, desc="Processing Columns", leave=True):
        sentences = chunk[column].fillna("").tolist()
        embeddings = generate_response_and_embed_batch_llama(sentences, llama_embedder, batch_size=batch_size)
        chunk.loc[:, f'emb_{column}'] = [json.dumps(emb.tolist()) if emb is not None else "" for emb in embeddings]
        tqdm.write(f"Processed column: {column}")
    return chunk

# Function to load data
def load_data(data_path):
    return pd.read_csv(data_path, sep='\t')

# Function to save the processed data
def save_data(processed_chunk, output_file):
    processed_chunk.to_csv(output_file, sep='\t', index=False)

# Main function to parse command-line arguments and run the script
def main():
    parser = argparse.ArgumentParser(description='Run LLaMA embeddings for a given dataset')
    parser.add_argument('--input', type=str, required=True, help='Path to input TSV file')
    parser.add_argument('--output', type=str, required=True, help='Path to save output TSV file')
    parser.add_argument('--model_path', type=str, default='meta-llama/Meta-Llama-3.1-8B', help='Hugging Face model path')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for embedding')
    parser.add_argument('--max_length', type=int, default=9200, help='Maximum sequence length')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the model on')
    
    args = parser.parse_args()

    # Load the data
    data = load_data(args.input)
    
    # Initialize the LLaMA embedding model
    llama_embedder = LlamaSentenceEmbedding(model_path=args.model_path, device=args.device, max_length=args.max_length)
    
    # Process the data chunk
    processed_chunk = process_chunk_llama(data, llama_embedder, batch_size=args.batch_size)
    
    # Save the processed data
    save_data(processed_chunk, args.output)

    print(f"Processing completed. Data saved to {args.output}")

if __name__ == "__main__":
    main()
