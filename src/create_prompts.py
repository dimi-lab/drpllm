import pandas as pd
import random
import string
import argparse

def generate_random_string(length=5):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string

def load_dataset(input_path):
    return pd.read_csv(input_path, sep='\t', compression='gzip')

def generate_ccl_prompts(data_df):
    def generate_question(row):
        return f"Is cell line {row['cell_line_name']} Resistant or Sensitive to drug {row['drug_name']}?\n"
    
    def generate_refined_prompt_context(row):
        return f"You are an expert scientist, your task is to return a single word answer: Resistant or Sensitive.\nQuery: {row['question']} Context: {row['CONTEXT']}\nAnswer:"

    def generate_refined_prompt_drug(row):
        return f"You are an expert scientist, your task is to return a single word answer: Resistant or Sensitive.\nQuery: {row['question']} Context: {row['drug_final_description']}\nAnswer:"

    def generate_refined_prompt_cell(row):
        return f"You are an expert scientist, your task is to return a single word answer: Resistant or Sensitive.\nQuery: {row['question']} Context: {row['cellline_description']}\nAnswer:"

    data_df['question'] = data_df.apply(generate_question, axis=1)
    data_df['refined_prompt_context'] = data_df.apply(generate_refined_prompt_context, axis=1)
    data_df['refined_prompt_drug'] = data_df.apply(generate_refined_prompt_drug, axis=1)
    data_df['refined_prompt_cell'] = data_df.apply(generate_refined_prompt_cell, axis=1)

    return data_df

def generate_pdx_prompts(data_df):
    def generate_question(row):
        return f"Is model {row['Model']} Resistant or Sensitive to drug {row['Treatment']}?\n"

    def generate_refined_prompt_context(row):
        return f"You are an expert scientist, your task is to return a single word answer: Resistantor Sensitive.\nQuery: {row['question']} Context: {row['CONTEXT']}\nAnswer:"

    def generate_refined_prompt_drug(row):
        return f"You are an expert scientist, your task is to return a single word answer: Resistantor Sensitive.\nQuery: {row['question']} Context: {row['drug_desc']}\nAnswer:"

    def generate_refined_prompt_cell(row):
        return f"You are an expert scientist, your task is to return a single word answer: Resistantor Sensitive.\nQuery: {row['question']} Context: {row['Cellline_desc']}\nAnswer:"
    data_df['question'] = data_df.apply(generate_question, axis=1)
    data_df['refined_prompt_context'] = data_df.apply(generate_refined_prompt_context, axis=1)
    data_df['refined_prompt_drug'] = data_df.apply(generate_refined_prompt_drug, axis=1)
    data_df['refined_prompt_cell'] = data_df.apply(generate_refined_prompt_cell, axis=1)

    return data_df

def save_dataset(data_df, output_path):
    data_df.to_csv(output_path, sep='\t', index=False)


def main():
    parser = argparse.ArgumentParser(description="Process and generate refined prompts for LLM embeddings.")
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input dataset TSV file.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the processed output TSV file.')
    parser.add_argument('--dataset', type=str, required=True, help='ccl or pdx')
    args = parser.parse_args()
    data_df = load_dataset(args.input_path)
    if args.dataset == 'ccl':
        data_df = generate_ccl_prompts(data_df)
        save_dataset(data_df, args.output_path)
        print(f"Processing complete. Dataset saved to {args.output_path}")
    else:
        data_df = generate_pdx_prompts(data_df)
        save_dataset(data_df, args.output_path)
        print(f"Processing complete. Dataset saved to {args.output_path}")

if __name__ == "__main__":
    main()
