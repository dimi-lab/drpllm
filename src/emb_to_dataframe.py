import os
import glob
import pandas as pd
import ast
from tqdm import tqdm

def load_data(files_pattern, ccle_file):
    """
    Load the data from multiple TSV files and concatenate them with a CCLE dataset.

    Parameters:
    - files_pattern: The pattern for finding the processed chunk TSV files.
    - ccle_file: The file path for the CCLE dataset.

    Returns:
    - combined_df: A DataFrame containing the concatenated data.
    """
    file_list = glob.glob(files_pattern)
    df_list = [pd.read_csv(file, sep='\t') for file in file_list]
    gdsc_df = pd.concat(df_list, ignore_index=True)
    
    ccle_df = pd.read_csv(ccle_file, sep='\t')
    
    combined_df = pd.concat([ccle_df, gdsc_df], axis=0)
    
    return combined_df

def process_and_save_embedded_df(df, emb_column, output_filename):
    """
    Converts a string column containing arrays into a DataFrame, merges it with AUC and label,
    and saves it to a CSV file.

    Parameters:
    - df: The DataFrame containing the embedding data.
    - emb_column: The column name that contains the string representation of the embeddings.
    - output_filename: The name of the output CSV file.
    """
    # Parse the string column into a list of arrays
    emb_list = [ast.literal_eval(x) for x in tqdm(df[emb_column], desc=f"Processing {emb_column}")]

    # Convert the list of arrays into a DataFrame
    emb_df = pd.DataFrame(emb_list)

    # Rename columns for clarity
    emb_df.columns = [f'feature_{i+1}' for i in range(emb_df.shape[1])]

    # Add AUC, label, and cancer_type columns back to the new DataFrame
    emb_df['AUC'] = df['AUC']
    emb_df['label'] = df['label']
    emb_df['cancer_type'] = df['cancer_type']

    # Save the resulting DataFrame to a CSV file
    emb_df.to_csv(output_filename, index=False)
    print(f"Saved to {output_filename}")

def main():
    # Define file paths
    files_pattern = '/projects/wangc/rohan/hackathon/src/data_processed_complete/processed_chunk_*.tsv'
    ccle_file = '/projects/wangc/rohan/hackathon/src/CCLE_gemma2b_emb_prompt.hidden_state.merge.tsv'
    
    # Load and combine datasets
    combined_df = load_data(files_pattern, ccle_file)
    
    # Define specific DataFrames for each embedding type
    question_emb_df = combined_df[['AUC', 'label', 'cancer_type', 'emb_question']]
    question_0_emb_df = combined_df[['AUC', 'label', 'cancer_type', 'emb_question0']]
    few_shot_emb_df = combined_df[['AUC', 'label', 'cancer_type', 'emb_refined_prompt_few_shot']]
    context_emb_df = combined_df[['AUC', 'label', 'cancer_type', 'emb_refined_prompt_context']]
    cellline_context_emb_df = combined_df[['AUC', 'label', 'cancer_type', 'emb_refined_prompt_cell']]
    drug_context_emb_df = combined_df[['AUC', 'label', 'cancer_type', 'emb_refined_prompt_drug']]

    # Process and save embeddings to CSV files
    process_and_save_embedded_df(question_emb_df, 'emb_question', 'question_emb_data.csv')
    process_and_save_embedded_df(question_0_emb_df, 'emb_question0', 'question0_emb_data.csv')
    process_and_save_embedded_df(few_shot_emb_df, 'emb_refined_prompt_few_shot', 'few_shot_emb_data.csv')
    process_and_save_embedded_df(context_emb_df, 'emb_refined_prompt_context', 'context_emb_data.csv')
    process_and_save_embedded_df(cellline_context_emb_df, 'emb_refined_prompt_cell', 'cellline_context_emb_data.csv')
    process_and_save_embedded_df(drug_context_emb_df, 'emb_refined_prompt_drug', 'drug_context_emb_data.csv')

if __name__ == '__main__':
    main()
