import pandas as pd
import random
import string
import argparse

# Function to generate a random string
def generate_random_string(length=5):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string

# Function to load dataset
def load_dataset(input_path):
    return pd.read_csv(input_path, sep='\t')

# Function to process the dataset
def process_dataset(data_df):
    data_df['question'] = "Is the drug " + data_df['NAME']  + " Resistant or Sensitive to cell line " + data_df['CELL_LINE'] + "?\n"
    data_df['question_0'] = "Is the drug " + generate_random_string()  + " Resistant or Sensitive to cell line " + generate_random_string() + "?\n"

    data_df['question_1'] = "Is the drug " + data_df['NAME']  + " Resistant or Sensitive to cell line " + generate_random_string() + "?\n"
    
    data_df['question_2'] = "Is the drug " + generate_random_string()  + " Resistant or Sensitive to cell line " + data_df['CELL_LINE'] + "?\n"
    prefix = 'You are an expert scientist. I will give the query, and your task is to return a single word answer: Resistant or Sensitive.\n'
    data_df['question_prompt'] = prefix + data_df['question']
    data_df['question_prompt_0'] = prefix + data_df['question_0']
    data_df['question_prompt_1'] = prefix + data_df['question_1']
    data_df['question_prompt_2'] = prefix + data_df['question_2']    
    data_req_df = data_df[['auc', 'auc_disc', 'CELL_LINE', 'cancer_type', 'cell_line_description',
                           'NAME', 'drug_description', 'SMILE', 'CONTEXT', 'question', 'question_0',
                           'question_1', 'question_2', 'question_prompt', 'question_prompt_0',
                           'question_prompt_1', 'question_prompt_2']]

    # Helper function to create cell and drug description sentences
    def create_cell_sentence(row):
        return f"The cell line {row['CELL_LINE']}, which is taken from {row['cell_line_description']}"

    def create_drug_sentence(row):
        return f" {row['drug_description']}. {row['SMILE']} "

    data_req_df['cellline_description'] = data_req_df.apply(create_cell_sentence, axis=1)
    data_req_df['drug_final_description'] = data_req_df.apply(create_drug_sentence, axis=1)

    data_req_df['DATASET'] = "DATASET"
    data_req_df = data_req_df.drop_duplicates().reset_index(drop=True)

    return data_req_df[['cellline_description', 'drug_final_description', 'question', 'question_0',
                        'question_1', 'question_2','question_prompt', 'question_prompt_0',
                        'question_prompt_1', 'question_prompt_2','auc', 'auc_disc', 'CELL_LINE',
                        'cancer_type', 'NAME', 'DATASET']]

# Function to create refined prompts
def generate_refined_prompts(data_df):
    def generate_refined_prompt(row):
        few_shots = (
            "Example 1:\n"
            "Query: Is the drug Topotecan sensitive or resistant to 22Rv1 cell line?\n"
            "Answer: Resistant\n"
            "Example 2:\n"
            "Query: Is the drug jq1 sensitive or resistant to cell line KMS-11?\n"
            "Answer: Sensitive\n"
        )
        return (
            f"You are an expert scientist. I will give you two examples, followed by the query and the context, and your task is to return a single word answer: Resistant or Sensitive.\n"
            f"{few_shots}"
            f"New Query: {row['question']}\n"
            f"Context: {row['CONTEXT']}\n"
            f"Answer:"
        )

    def generate_refined_prompt_context(row):
        return f"You are an expert scientist, your task is to return a single word answer: Resistant or Sensitive.\nQuery: {row['question']} Context: {row['CONTEXT']}\nAnswer:"

    def generate_refined_prompt_drug(row):
        return f"You are an expert scientist, your task is to return a single word answer: Resistant or Sensitive.\nQuery: {row['question']} Context: {row['drug_final_description']}\nAnswer:"

    def generate_refined_prompt_cell(row):
        return f"You are an expert scientist, your task is to return a single word answer: Resistant or Sensitive.\nQuery: {row['question']} Context: {row['cellline_description']}\nAnswer:"

    # Apply refined prompt generation
    data_df['refined_prompt_few_shot'] = data_df.apply(generate_refined_prompt, axis=1)
    data_df['refined_prompt_context'] = data_df.apply(generate_refined_prompt_context, axis=1)
    data_df['refined_prompt_drug'] = data_df.apply(generate_refined_prompt_drug, axis=1)
    data_df['refined_prompt_cell'] = data_df.apply(generate_refined_prompt_cell, axis=1)

    return data_df

# Function to save the processed dataset
def save_dataset(data_df, output_path):
    data_df.to_csv(output_path, sep='\t', index=False)

# Main function for command-line execution
def main():
    parser = argparse.ArgumentParser(description="Process and generate refined prompts for LLM embeddings.")
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input dataset TSV file.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the processed output TSV file.')

    args = parser.parse_args()

    # Load dataset
    data_df = load_dataset(args.input_path)

    # Process the dataset
    data_req_df = process_dataset(data_df)

    # Generate refined prompts
    data_req_df['CONTEXT'] = data_req_df['cellline_description'] + "." + data_req_df['drug_final_description']
    data_req_df = generate_refined_prompts(data_req_df)

    # Save the processed dataset
    save_dataset(data_req_df, args.output_path)

    print(f"Processing complete. Dataset saved to {args.output_path}")

if __name__ == "__main__":
    main()
