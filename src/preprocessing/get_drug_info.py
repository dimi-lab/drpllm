import os.path
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import requests
import json
import time
from jsonpath_ng import jsonpath, parse
import argparse

def get_CID(drug_symbol):
    try:
        # drug_symbol = "dacarbazine"
        # drug_symbol =  "cetuximab"
        URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/" + drug_symbol + "/cids/TXT"
        r = requests.get(url=URL)
        if r.status_code != 200:
            return ""
        else:
            cid_lines = r.text.strip().split("\n")
            return cid_lines[0]
    except:
        return 'get_CID Did not work'


def get_SMILES(cid):
    try:
        URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/" + cid + "/property/CanonicalSMILES/TXT"
        r = requests.get(url=URL)
        if r.status_code != 200:
            return ""
        else:
            return r.text.strip()
    except:
        return 'get_SMILES did not work'


def get_description(cid):
    try:
        URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/" + cid + "/description/Json"
        r = requests.get(url=URL)
        des_dict = json.loads(r.text.strip())
        des = des_dict["InformationList"]["Information"][1]["Description"]
        if r.status_code != 200:
            return ""
        else:
            return des
    except:
        return 'get_description did not work'

def get_pubchem_title(cid):
    try:
        URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/" + cid + "/property/Title/TXT"
        r = requests.get(url=URL)
        if r.status_code != 200:
            return ""
        else:
            return r.text.strip()
    except:
        return 'get_raw_page did not work'

def get_SourceName(json_data, ReferenceNumber):
    jsonpath_expression = parse('Record[*].Reference[*]')
    for match in jsonpath_expression.find(json_data):
        ref = match.value
        if ref["ReferenceNumber"] == ReferenceNumber:
            return ref["SourceName"]


def get_Description(CID):
    request_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/%s/JSON?heading=Names+and+Identifiers" % CID
    json_html = urlopen(request_url)
    json_data = json.load(json_html)
    j_exp_sec = parse('Record[*].Section[*]')
    descriptions = {}
    for section_match in j_exp_sec.find(json_data):
        rec = section_match.value
        a = rec["Section"]
        for aa in a:
            if aa["TOCHeading"] == "Record Description":
                d = aa["Information"]
                for dd in d:
                    r_num = dd["ReferenceNumber"]
                    is_see_also = False
                    if "Name" in dd:
                        if dd["Name"] == "See Also":
                            is_see_also = True
                    if not is_see_also:
                        des = dd["Value"]["StringWithMarkup"][0]["String"]
                        source_name = get_SourceName(json_data, r_num)
                        descriptions[source_name] = des
                        # print("%s: %s" % (source_name, des))
    return descriptions



def process_drug_list(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    # Read drug names from input file
    with open(input_file, "r") as file:
        drug_names = [line.strip() for line in file if line.strip()]

    results = []

    print(f"Processing {len(drug_names)} drugs...")

    for drug in drug_names:
        print(f"🔹 Fetching data for: {drug}")

        cid = get_CID(drug)
        smiles = get_SMILES(cid) if cid != "Not Available" else "Not Available"
        description = get_description(cid) if cid != "Not Available" else "Not Available"
        title = get_pubchem_title(cid) if cid != "Not Available" else "Not Available"

        results.append({
            "Drug Name": drug,
            "PubChem CID": cid,
            "PubChem Title": title,
            "SMILES": smiles,
            "Description": description
        })

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

    print(f"\n✅ Processing complete. Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Fetch PubChem data for a list of drugs.")
    parser.add_argument("--input", type=str, required=True, help="Path to input file containing drug names (one per line).")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output CSV file.")

    args = parser.parse_args()

    process_drug_list(args.input, args.output)

if __name__ == "__main__":
    main()
