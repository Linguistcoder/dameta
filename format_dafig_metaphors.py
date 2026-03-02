import pdb
import os
import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
pd.set_option('display.max_colwidth', None)

DATA_DIR = "../data"

### Load files

# Load csv files 'mtp_lemmatized_main' and 'mtp_lemmatized_reanno' from DATA_DIR
mtp_main_df = pd.read_csv(os.path.join(DATA_DIR, 'mtp_sentences_lemmatized_main.csv'))
mtp_consensus_df = pd.read_csv(os.path.join(DATA_DIR,'mtp_sentences_lemmatized_consensus.csv'))
mtp_reanno_df = pd.read_csv(os.path.join(DATA_DIR,'mtp_sentences_lemmatized_reanno.csv'))

# Pull metaphor list from txt-file
metaphor_list = []
with open(os.path.join(DATA_DIR, 'metaphor_lemma_list.txt'), 'r', encoding='utf-8') as f:
    for line in f:
        # Make sure that numbers and other non/alphanumeric characters are removed before adding it to the list
        metaphor_list.append(re.sub(r'[^a-zA-Z0-9 ]+', '', line.strip()))

### Data cleaning ###

# Insert a column named 'comments' with the name of the dataset ('single-annotated' for 'main' and 'double-annotated' for CONSENSUS and REANNO), and combine all dataframes
mtp_df = pd.concat([
    mtp_main_df.assign(source='single-annotated'),
    mtp_consensus_df.assign(source='double-annotated'),
    mtp_reanno_df.assign(source='double-annotated')
], ignore_index=True)

# Print shape of the combined dataframe
print(mtp_df.shape)

# Make sure lemma is not case sensitive
mtp_df['lemma'] = mtp_df['lemma'].str.lower()

# Select only rows where type is 'single_word' and POS tag is either NOUN, VERB, or ADJ
mtp_df = mtp_df[
    (mtp_df['type'] == 'single_word') & 
    (
        (mtp_df['pos'].isin(['NOUN', 'VERB']) & (mtp_df['lemma'].str.len() >= 5)) |
        ((mtp_df['pos'] == 'ADJ') & (mtp_df['lemma'].str.len() >= 5))
    )
]
# Print shape again
print(mtp_df.shape)

# Make another column called met_type, which contains 1 if the lemma is in the metaphor list, and 0 otherwise
mtp_df['met_type'] = mtp_df['lemma'].apply(lambda x: '(1)' if x in metaphor_list else '(2)')

# Update the value of met_type to 3 if the value of Conventionality column is NOV
mtp_df.loc[mtp_df['Conventionality'] == 'NOV', ['met_type']] = '(3)'

# Remove sentences that does not end with period
mtp_df = mtp_df[mtp_df['sentence'].str.endswith('.')]

#TODO: Correct the one error in the data where the values for directness and conventionality are switched accidentally

### Transforming into benchmark data format ###

#Create empty columns for DDO_entry, unique, exp1, exp2, epx3, epx4
mtp_df['DDO_entry'] = ''
mtp_df['unique'] = ''
mtp_df['exp1'] = ''
mtp_df['exp2'] = ''
mtp_df['exp3'] = ''
mtp_df['exp4'] = ''
mtp_df['comments'] = ''

# Reorder columns in met_df as follows: lemma, met_type, DDO_entry, unique, exp1, exp2, exp3, exp4
mtp_df = mtp_df[['lemma', 'met_type', 'DDO_entry', 'unique', 'sentence', 'exp1', 'exp2', 'exp3', 'exp4', 'source', 'comments']]

### Combine with another sheet ###

# Load existing benchmark data (from dafig file)
existing_df = pd.read_csv(os.path.join(DATA_DIR, 'gsheets_danish_met_bench_dafig.tsv'), sep='\t')

# Make sure lemma is lowercase in both
existing_df['lemma'] = existing_df['lemma'].str.lower()

# Merge the two dataframes based on sentence
existing_df = existing_df.merge(
    mtp_df[['sentence', 'met_type']], 
    how='left', 
    on='sentence',
    suffixes=('', '_new')
)

# Update met_type where we have new data
existing_df['met_type'] = existing_df['met_type_new'].fillna(existing_df['met_type'])
existing_df = existing_df.drop('met_type_new', axis=1)
# Get new sentences that don't exist in existing_df
new_sentences = mtp_df[~mtp_df['sentence'].isin(existing_df['sentence'])]

# Combine everything
combined_df = pd.concat([existing_df, new_sentences], ignore_index=True)

# Now handle duplicates more carefully
def keep_best_row(group):
    # First, get the met_type from any row in the group that has it
    met_type_val = group['met_type'].dropna().iloc[0] if not group['met_type'].dropna().empty else None

    # If any row has exp1 filled, keep that one
    filled_exp = group[group['exp1'].notna() & (group['exp1'] != '')]
    if not filled_exp.empty:
        best_row = filled_exp.iloc[0].copy()  # Make a copy to avoid SettingWithCopyWarning
    else:
        # Otherwise keep the shortest sentence (assuming it's cleaner)
        best_row = group.loc[group['sentence'].str.len().idxmin()].copy()
    
    # Now explicitly set the met_type on the selected row
    best_row['met_type'] = met_type_val
    return best_row

# Group by lemma and apply our logic
combined_df = combined_df.groupby('lemma').apply(keep_best_row).reset_index(drop=True)

# Save the result
combined_df.to_csv(os.path.join(DATA_DIR, 'updated_data.tsv'), sep='\t', index=False)

print("All done!")
