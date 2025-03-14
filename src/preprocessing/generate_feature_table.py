import os
import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import scale

def process_benchmark_data(expression_file, cnv_file, geneset_file):
    """
    Process gene expression and CNV data to extract top up/down-regulated genes and copy number variations.
    """
    # Read and scale gene expression data
    benchmark_data_exp = pd.read_csv(expression_file, sep='\t', skiprows=2)
    benchmark_data_exp.set_index(benchmark_data_exp.columns[0], inplace=True)
    benchmark_data_exp_scaled = pd.DataFrame(scale(benchmark_data_exp),
                                             index=benchmark_data_exp.index, columns=benchmark_data_exp.columns)
    # Read gene list and filter
    genesets = pd.read_csv(geneset_file, sep='\t')
    genesets = genesets[genesets['MSK-IMPACT'] == 'Yes']['Hugo Symbol']

    # Select scaled data based on geneset
    benchmark_data_exp_scaled_sel = benchmark_data_exp_scaled.loc[:, benchmark_data_exp_scaled.columns.isin(genesets)]
    
    # Get top upregulated and downregulated genes
    ngenes = 10
    gene_list_exp_up = benchmark_data_exp_scaled_sel.apply(lambda x: ','.join(benchmark_data_exp_scaled_sel.columns[x.argsort()[-ngenes:]]), axis=1)
    gene_list_exp_down = benchmark_data_exp_scaled_sel.apply(lambda x: ','.join(benchmark_data_exp_scaled_sel.columns[x.argsort()[:ngenes]]), axis=1)

    top_genes = pd.DataFrame({'cell_id': benchmark_data_exp_scaled_sel.index, 'gene_list_exp_up': gene_list_exp_up, 'gene_list_exp_down': gene_list_exp_down})

    # Process CNV data
    benchmark_data_cnv = pd.read_csv(cnv_file, sep='\t')
    benchmark_data_cnv.set_index(benchmark_data_cnv.columns[0], inplace=True)
    benchmark_data_cnv_sel = benchmark_data_cnv.loc[:, benchmark_data_cnv.columns.isin(genesets)]

    # Get list of genes with gains and losses
    gene_list_cnv_loss = benchmark_data_cnv_sel.apply(lambda x: ','.join(benchmark_data_cnv_sel.columns[x.argsort()[:2]]), axis=1)
    gene_list_cnv_gain = benchmark_data_cnv_sel.apply(lambda x: ','.join(benchmark_data_cnv_sel.columns[x.argsort()[-2:]]), axis=1)

    top_genes_cnv = pd.DataFrame({'cell_id': benchmark_data_cnv.index, 'gene_list_cnv_gain': gene_list_cnv_gain, 'gene_list_cnv_loss': gene_list_cnv_loss})

    data_exp_cnv = pd.merge(top_genes, top_genes_cnv, on='cell_id', how='outer')
    return data_exp_cnv

def create_cellline_sentence(row):
    """Generate a descriptive sentence for a cell line."""
    return (
        f"The cell line {row.get('CELL_LINE_NAME', 'UNKNOWN')}, with Sanger model ID {row.get('SANGER_MODEL_ID', 'UNKNOWN')}, "
        f"originates from the dataset {row.get('DATASET', 'UNKNOWN')}. It was derived from tissue site {row.get('Site', 'UNKNOWN')}, "
        f"with a pathology described as {row.get('Histology', 'UNKNOWN')}. Upregulated genes include {row.get('gene_list_exp_up', 'UNKNOWN')}, "
        f"while downregulated genes include {row.get('gene_list_exp_down', 'UNKNOWN')}. Genes with copy number gain include {row.get('gene_list_cnv_gain', 'UNKNOWN')}, "
        f"and genes with a copy number loss include {row.get('gene_list_cnv_loss', 'UNKNOWN')}."
    )

def create_drug_sentence(row):
    """Generate a descriptive sentence for a drug."""
    return (
        f"The drug {row.get('DRUG_NAME', 'UNKNOWN')} targets {row.get('PUTATIVE_TARGET', 'UNKNOWN')} and is studied in the {row.get('DATASET', 'UNKNOWN')} dataset. "
        f"It has a chemical formula of {row.get('formula', 'UNKNOWN')} and a molecular weight of {row.get('weight', 'UNKNOWN')} g/mol. "
        f"The SMILES string for this drug is {row.get('canSMILES', 'UNKNOWN')}."
    )

def create_question_sentence(row):
    """Generate a question sentence for drug sensitivity."""
    return f"Is the cell line {row.get('CELL_LINE_NAME', 'UNKNOWN')} sensitive or resistant to the drug {row.get('DRUG_NAME', 'UNKNOWN')}? "

def convert_string_columns_to_lowercase(df):
    """
    Convert all string columns in a DataFrame to lowercase.
    """
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.lower()
    return df

def main():
    parser = argparse.ArgumentParser(description="Process gene expression, CNV data, and generate descriptive sentences.")
    parser.add_argument("--expression", type=str, required=True, help="Path to gene expression file.")
    parser.add_argument("--cnv", type=str, required=True, help="Path to CNV data file.")
    parser.add_argument("--geneset", type=str, required=True, help="Path to gene set file.")
    parser.add_argument("--cellline_info", type=str, required=True, help="Path to cell line information file.")
    parser.add_argument("--gdsc_data", type=str, required=True, help="Path to GDSCv2 data file.")
    parser.add_argument("--drug_data", type=str, required=True, help="Path to drug data file.")
    parser.add_argument("--cosmic_tissue", type=str, required=True, help="Path to COSMIC tissue classification file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the processed output CSV.")

    args = parser.parse_args()

    # Load datasets
    cell_line_map_df = pd.read_csv(args.cellline_info, sep='\t')
    drug_data_df = pd.read_csv(args.drug_data, sep='\t')
    gdsc_data_df = pd.read_csv(args.gdsc_data, sep='\t')
    cosmic_tissue_df = pd.read_csv(args.cosmic_tissue, sep='\t')

     # Convert string columns to lowercase
#    cell_line_map_df = convert_string_columns_to_lowercase(cell_line_map_df)
    drug_data_df = convert_string_columns_to_lowercase(drug_data_df)
    gdsc_data_df = convert_string_columns_to_lowercase(gdsc_data_df)
    cosmic_tissue_df = convert_string_columns_to_lowercase(cosmic_tissue_df)
    exp_cnv_df = process_benchmark_data(args.expression, args.cnv, args.geneset)
    cell_line_exp_cnv_df = pd.merge(exp_cnv_df, cell_line_map_df, left_on='cell_id', right_on='improve_sample_id', how='left')
    gdsc_data_cellline_df = pd.merge(gdsc_data_df, cell_line_map_df, left_on='CELL_LINE_NAME', right_on='common_name', how='left')
    gdsc_data_cellline_df = pd.merge(gdsc_data_cellline_df, drug_data_df, left_on='DRUG_NAME', right_on='chem_name', how='left')
    gdsc_data_cellline_df.to_csv('gdsc_data_celline.tsv', sep='\t', index=None)
    exp_cnv_df.to_csv('exp_cnv_data.tsv', sep='\t', index=None)
    gdsc_data_cellline_df = pd.merge(gdsc_data_cellline_df, cell_line_exp_cnv_df, left_on='improve_sample_id',
                                     right_on='cell_id', how='left')
    gdsc_data_cellline_df = gdsc_data_cellline_df.loc[:, ~gdsc_data_cellline_df.columns.str.endswith('_y')]
    gdsc_data_cellline_df.columns = gdsc_data_cellline_df.columns.str.replace('_x$', '', regex=True)
    gdsc_data_cellline_df['COSMIC_ID'] = gdsc_data_cellline_df['COSMIC_ID'].astype(str)
    cosmic_tissue_df['COSMIC_ID'] = cosmic_tissue_df['COSMIC_ID'].astype(str)
    gdsc_data_cellline_df = pd.merge(
        gdsc_data_cellline_df,
        cosmic_tissue_df[['COSMIC_ID', 'Line', 'Site', 'Histology']],
        on='COSMIC_ID',
        how='left')

    # Generate descriptive sentences
    gdsc_data_cellline_df['label'] = gdsc_data_cellline_df['AUC'].apply(lambda x: 'Resistant' if x > 0.5 else 'Sensitive')
    gdsc_data_cellline_df['Cell_line_desc'] = gdsc_data_cellline_df.apply(create_cellline_sentence, axis=1)
    gdsc_data_cellline_df['drug_desc'] = gdsc_data_cellline_df.apply(create_drug_sentence, axis=1)
    gdsc_data_cellline_df['question'] = gdsc_data_cellline_df.apply(create_question_sentence, axis=1)
#    print(gdsc_data_cellline_df[~gdsc_data_cellline_df['gene_list_exp_up'].isna()])
    # Save the processed data
    gdsc_data_cellline_df.to_csv(args.output, index=False)

    print(f"\n✅ Process completed. Results saved to {args.output}")

if __name__ == "__main__":
    main()
