import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ast
import re
import sys
from rdkit import Chem
from rdkit.Chem import Draw
import requests
import pubchempy as pcp
from pubchempy import get_compounds, PubChemHTTPError
from pubchempy import BadRequestError
import logging
logging.getLogger("pubchempy").setLevel(logging.WARNING)
import argparse



def main(input_file, output_file):
    #run_pubchempy(input_file, output_file)
    smiles_dict = get_smiles(input_file)
    df = pd.DataFrame(list(smiles_dict.items()), columns=['Drug', 'SMILES'])
    df.to_csv(output_file, index=None)


def get_chembl_id_from_smiles(smiles):
    try:
        # Search for the compound by SMILES
        compound = pcp.get_compounds(smiles, 'smiles', record_type='3d')[0]
        
        # Get the ChEMBL ID from the compound
        chembl_id = compound.to_dict(properties=['chembl_id'])['chembl_id']
        
        return chembl_id
    except IndexError:
        return None
    
def get_smiles_from_name(chemical_name):
    smile_dict = {}
    try:
        compound = pcp.get_compounds(chemical_name, 'name')
        for result in compound:
            smile_string = result.isomeric_smiles
            if smile_string:
                return smile_string
            else:
                return None
    except BadRequestError:
        print("Error: Bad request to PubChem API")
    return smile_dict


def get_name_from_smiles(smiles_string):
    try:
        compounds = pcp.get_compounds(smiles_string, 'smiles')
    #print(compounds.ID)
        if compounds:
            cid = [comp.cid for comp in compounds]
            return cid[0]
        else:
            print("compount not found")
            return None
    except BadRequestError:
        return None


def drug_features(drug_file, interaction_file):
    df = pd.read_feather(drug_file)
    in_df = pd.read_csv(interaction_file, sep='\t')
    in_df = in_df[~in_df['drug_name'].isna()]
    in_df['chembl'] = in_df['drug_concept_id'].str.replace('chembl:', "")
    in_df = in_df[~in_df['chembl'].isna()]
    drug_name_list = list(set(in_df['drug_claim_name'].str.upper().to_list()))
    drug_name_df = pd.DataFrame(drug_name_list)
    drug_name_df.to_csv('drugname2ind.tsv', sep='\t')
    return drug_name_list

def get_smiles(drug_list):
    smiles_dict = {}
    with open(drug_list) as fin:
        for i in fin:
            smile_string = i.strip().split(",")[1]
            smiles = get_smiles_from_name(smile_string)
            if smiles:
                smiles_dict[smile_string] = smiles
            else:
                smiles_dict[smile_string] = None
    return smiles_dict

def run_pubchempy(drug_file, outfile):
    df = pd.read_csv(drug_file, header=None, sep='\t')
    df.columns = ["DRUG_NAME"]
    df['SMILES'] = df['DRUG_NAME'].apply(lambda x: get_smiles_from_name(x))
    df.reset_index(drop=True) 
    df.to_csv(outfile, index=None)


if __name__ == "__main__":
    import argparse
    import sys
    drug_file = sys.argv[1]
    out_file = sys.argv[2]
    main(drug_file, out_file)
    

