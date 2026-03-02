### Script to make multiple choice examples from human evaluation ###

import yaml
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import pdb

# Helper functions
def shuffle_explanations(row: pd.Series, seed: int = None) -> Tuple[Dict, Dict]:
    """Shuffle explanations and return mapping"""
    if seed is not None:
        np.random.seed(seed)
    
    # Original explanations with their labels
    original_explanations = {
        'exp1': row['exp1'],
        'exp2': row['exp2'], 
        'exp3': row['exp3'],
        'exp4': row['exp4']
    }
    
    # Create list of (original_key, explanation) pairs
    exp_pairs = list(original_explanations.items())
    
    # Shuffle the pairs
    np.random.shuffle(exp_pairs)
    
    # Create shuffled dict and mapping
    shuffled = {}
    reverse_mapping = {}  # Maps from shuffled position to original position
    
    for i, (original_key, explanation) in enumerate(exp_pairs):
        shuffled_key = f'exp{i+1}'
        shuffled[shuffled_key] = explanation
        reverse_mapping[shuffled_key] = original_key
    
    return shuffled, reverse_mapping

# Create a DataFrame to store results - NOW WITH SOURCE TRACKING
results = pd.DataFrame(columns=[
    'question_id',      # NEW: Sequential ID for the survey
    'source_dataset',   # NEW: Which dataset this came from
    'source_index',     # NEW: Original index in source file
    'question', 
    'exp1', 'exp2', 'exp3', 'exp4', 
    'answer'
])

# Read config.yaml file (in same folder)
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

### Prepare data ###

# Read dataset dict from config
datasets = config['datasets']

## Loop over each dataset, read in the data from file_path, extract examples

# Map dataset name to number of examples
example_counts = {
    'dafig': 20,
    'ddo': 20,
    'ad_hoc': 8,
    'unik': 2
}

# Global counter for question IDs
question_id = 1

# Loop over each dataset
for dataset_config in config['datasets']:
    # Read dataset
    name = dataset_config['name']
    file_path = dataset_config['file_path']
    df = pd.read_csv(file_path, sep='\t')
    
    # drop rows with missing explanations (exp1, exp2, exp3, exp4)
    df = df.dropna(subset=['exp1', 'exp2', 'exp3', 'exp4'])
    
    # Extract examples using the example_counts dictionary
    examples = df.sample(n=example_counts[name], random_state=42)
    
    # Create answer columns by shuffling the original explanations
    for source_idx, example in examples.iterrows():
        # Create question for THIS specific row
        results.loc[question_id - 1, 'question_id'] = question_id
        results.loc[question_id - 1, 'source_dataset'] = name
        results.loc[question_id - 1, 'source_index'] = source_idx  # Original index from source file
        results.loc[question_id - 1, 'question'] = f"Hvilke af disse fire parafraser beskriver bedst betydningen af ordet {example['lemma']} i følgende sætning: {example['sentence']}?"
        
        # Shuffle and add explanations to DataFrame
        shuffled, reverse_mapping = shuffle_explanations(example)
        forward_mapping = {v: k for k, v in reverse_mapping.items()}
        
        results.loc[question_id - 1, 'exp1'] = shuffled['exp1']
        results.loc[question_id - 1, 'exp2'] = shuffled['exp2']
        results.loc[question_id - 1, 'exp3'] = shuffled['exp3']
        results.loc[question_id - 1, 'exp4'] = shuffled['exp4']
        results.loc[question_id - 1, 'answer'] = forward_mapping['exp1']
        
        question_id += 1

# Save FULL results with provenance
results.to_csv('multiple_choice_metaphors_FULL.csv', index=False)

# Save SURVEY VERSION without source tracking columns
survey_version = results[['question', 'exp1', 'exp2', 'exp3', 'exp4']]
survey_version.to_csv('multiple_choice_metaphors_SURVEY.csv', index=False)

print(f"✓ Created {len(results)} questions")
print(f"✓ Full version: multiple_choice_metaphors_FULL.csv")
print(f"✓ Survey version: multiple_choice_metaphors_SURVEY.csv")