import argparse
import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process and merge CCLE & GDSC datasets.")
    parser.add_argument("--gdsc", type=str, required=True, help="Path to GDSC data file")
    parser.add_argument("--ccle", type=str, required=True, help="Path to CCLE data file")
    parser.add_argument("--cosmic", type=str, required=True, help="Path to COSMIC tissue classification file")
    parser.add_argument("--drug", type=str, required=True, help="Path to drug info file")
    parser.add_argument("--expression", type=str, required=True, help="Path to cancer gene expression file")
    parser.add_argument("--ccl_info", type=str, required=True, help="Path to CCL info file")
    parser.add_argument("--copy_number", type=str, required=True, help="Path to cancer copy number file")
    parser.add_argument("--output", type=str, default="CCLE_GDSCv2_metric_combined_data.feather", help="Output feather file")
    
    return parser.parse_args()

def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    """Convert SMILES to a Morgan fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fingerprint)
    return None

def load_and_prepare_data(args):
    """Load and preprocess all datasets."""
    # Load datasets
    gdsc_data_df = pd.read_csv(args.gdsc, sep='\t')
    ccle_data_df = pd.read_csv(args.ccle, sep='\t')
    cosmic_df = pd.read_csv(args.cosmic, sep='\t')
    drug_df = pd.read_csv(args.drug, sep='\t')
    expression_df = pd.read_csv(args.expression, sep='\t', skiprows=2).rename(columns={'Unnamed: 0': 'improve_sample_id'})
    ccl_info_df = pd.read_csv(args.ccl_info, sep='\t')[['cancer_type', 'other_names', 'improve_sample_id']]
    copy_number_df = pd.read_csv(args.copy_number, sep='\t').rename(columns={'Unnamed: 0': 'improve_sample_id'})

    # Process CCLE data
    ccle_data_df = ccle_data_df[['improve_sample_id', 'improve_chem_id', 'auc1']]
    ccle_data_df.columns = ['improve_sample_id', 'improve_chem_id', 'AUC']

    # Process drug data
    drug_df = drug_df.rename(columns={'chem_name': 'DRUG_NAME'})
    drug_df['DRUG_NAME'] = drug_df['DRUG_NAME'].str.upper()
    drug_df['fingerprint'] = drug_df['isoSMILES'].apply(smiles_to_fingerprint)

    drug_keep_df = drug_df[['DRUG_NAME', 'fingerprint']]
    fingerprint_df = pd.DataFrame(drug_keep_df['fingerprint'].tolist(), columns=[f'FP_{i}' for i in range(len(drug_keep_df['fingerprint'][0]))])
    drug_keep_df = pd.concat([drug_keep_df[['DRUG_NAME']], fingerprint_df], axis=1)

    # Process expression & copy number data
    expression_ccl_df = pd.merge(expression_df, ccl_info_df, on='improve_sample_id', how='left').drop_duplicates().drop(columns=['improve_sample_id'])
    copy_number_ccl_df = pd.merge(copy_number_df, ccl_info_df, on='improve_sample_id', how='left').drop_duplicates().drop(columns=['improve_sample_id'])

    return gdsc_data_df, ccle_data_df, cosmic_df, drug_df, ccl_info_df, copy_number_ccl_df, expression_ccl_df, drug_keep_df

def process_ccle_data(ccle_data_df, cosmic_df, drug_df, ccl_info_df, drug_keep_df, copy_number_ccl_df, expression_ccl_df):
    """Process CCLE dataset and merge relevant information."""
    ccle_data_keep_df = pd.merge(ccle_data_df, ccl_info_df, on='improve_sample_id', how='left')
    ccle_data_keep_df = pd.merge(ccle_data_keep_df, cosmic_df, left_on='other_names', right_on='Line')
    ccle_data_keep_df = ccle_data_keep_df[['improve_chem_id', 'AUC', 'Line', 'Site', 'Histology', 'cancer_type']]
    ccle_data_keep_df = pd.merge(ccle_data_keep_df, drug_df, on='improve_chem_id', how='left')
    ccle_data_keep_df = ccle_data_keep_df[['AUC', 'weight', 'isoSMILES', 'Line', 'Site', 'Histology', 'DRUG_NAME', 'cancer_type']]
    
    ccle_data_keep_df = ccle_data_keep_df.rename(columns={'Line': 'CELL_LINE_NAME'})
    ccle_data_keep_df['DRUG_NAME'] = ccle_data_keep_df['DRUG_NAME'].str.upper()
    
    ccle_data_keep_drug_df = pd.merge(ccle_data_keep_df, drug_keep_df, on='DRUG_NAME', how='left')
    ccle_data_keep_drug_df = ccle_data_keep_drug_df.groupby(['AUC', 'weight', 'isoSMILES', 'CELL_LINE_NAME', 'Site', 'Histology']).first().reset_index()
    ccle_data_keep_drug_df = pd.merge(ccle_data_keep_drug_df, copy_number_ccl_df, left_on='CELL_LINE_NAME', right_on='other_names', how='left')
    ccle_data_keep_drug_df = pd.merge(ccle_data_keep_drug_df, expression_ccl_df, left_on='CELL_LINE_NAME', right_on='other_names', how='left')
    
    return ccle_data_keep_drug_df.drop_duplicates()

def process_gdsc_data(gdsc_data_df, drug_df, ccl_info_df, cosmic_df, drug_keep_df, copy_number_ccl_df, expression_ccl_df):
    """Process GDSC dataset and merge relevant information."""
    keep_columns = ['AUC', 'CELL_LINE_NAME', 'DRUG_NAME']
    gdsc_data_keep_df = gdsc_data_df[keep_columns]
    
    gdsc_data_keep_df = pd.merge(gdsc_data_keep_df, ccl_info_df, left_on='CELL_LINE_NAME', right_on='other_names', how='left')
    gdsc_data_keep_df['DRUG_NAME'] = gdsc_data_keep_df['DRUG_NAME'].str.upper()
    gdsc_data_keep_df = pd.merge(gdsc_data_keep_df, drug_df, on='DRUG_NAME', how='left')
    gdsc_data_keep_df = pd.merge(gdsc_data_keep_df, cosmic_df, left_on='CELL_LINE_NAME', right_on='Line', how='left')
    gdsc_data_keep_df = gdsc_data_keep_df.groupby(['AUC', 'weight', 'isoSMILES', 'CELL_LINE_NAME', 'Site', 'Histology']).first().reset_index()
    gdsc_data_keep_df = pd.merge(gdsc_data_keep_df, drug_keep_df, on='DRUG_NAME', how='left')
    gdsc_data_keep_df = pd.merge(gdsc_data_keep_df, copy_number_ccl_df, left_on='CELL_LINE_NAME', right_on='other_names', how='left')
    gdsc_data_keep_df = pd.merge(gdsc_data_keep_df, expression_ccl_df, left_on='CELL_LINE_NAME', right_on='other_names', how='left')

    cols_to_drop = ['Line', 'COSMIC_ID', 'other_names_y', 'other_names_x']
    return gdsc_data_keep_df.drop(columns=cols_to_drop, axis=1)

def main():
    args = parse_arguments()
    gdsc_data_df, ccle_data_df, cosmic_df, drug_df, ccl_info_df, copy_number_ccl_df, expression_ccl_df, drug_keep_df = load_and_prepare_data(args)

    ccle_data = process_ccle_data(ccle_data_df, cosmic_df, drug_df, ccl_info_df, drug_keep_df, copy_number_ccl_df, expression_ccl_df)
    gdsc_data = process_gdsc_data(gdsc_data_df, drug_df, ccl_info_df, cosmic_df, drug_keep_df, copy_number_ccl_df, expression_ccl_df)

    combined_data = pd.concat([ccle_data, gdsc_data], ignore_index=True).drop_duplicates().reset_index(drop=True)
    combined_data.to_feather(args.output)
    print(f"Processed data saved to {args.output}")

if __name__ == "__main__":
    main()

    
