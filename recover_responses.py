import pdb
import pandas as pd
import yaml
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data_dir = Path("data/humans")
responses_df = pd.read_csv(data_dir / 'responses_shuffled.tsv', sep="\t")
options_df = pd.read_csv(data_dir / 'options_shuffled.tsv', sep="\t")

# Merge to get the shuffled options
merged_df = responses_df.merge(options_df, on='question_id', how='left')

# Load original datasets
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# BUILD LOOKUP DICTIONARY ONCE - O(n) setup, O(1) access
original_lookup = {}
for dataset_info in config['datasets']:
    name = dataset_info['name']
    path = dataset_info['file_path']
    df = pd.read_csv(path, sep="\t")
    
    for idx, row in df.iterrows():
        key = (name, idx)
        original_lookup[key] = [row['exp1'], row['exp2'], 
                                row['exp3'], row['exp4']]

# Process each row
results = []
for idx, row in merged_df.iterrows():
    dataset_name = row['source_dataset']
    source_idx = row['source_index']
    
    # O(1) LOOKUP instead of DataFrame iloc
    key = (dataset_name, source_idx)
    if key not in original_lookup:
        raise ValueError(f"No original data found for {key}")
    
    original_options = original_lookup[key]
    
    # Get THIS row's shuffled option order
    shuffled_options = [row['exp1'], row['exp2'], row['exp3'], row['exp4']]
    
    # Create mapping: shuffled position -> original position
    mapping = {}
    for shuffled_pos, shuffled_text in enumerate(shuffled_options, start=1):
        try:
            original_pos = original_options.index(shuffled_text) + 1
            mapping[shuffled_pos] = original_pos
        except ValueError:
            print(f"ERROR at question_id {row['question_id']}:")
            print(f"  Looking for: {repr(shuffled_text)}")
            print(f"  Available options: {[repr(o) for o in original_options]}")
            raise
    
    # Convert responses
    result_row = {
        'question_id': row['question_id'],
        'source_dataset': dataset_name,
        'source_index': source_idx,
        'exp1_original': original_options[0],
        'exp2_original': original_options[1],
        'exp3_original': original_options[2],
        'exp4_original': original_options[3]
    }
    
    for responder in ['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9']:
        response = row[responder]
        if pd.isna(response):
            result_row[f'{responder}_original'] = None
        else:
            shuffled_pos = int(response)
            result_row[f'{responder}_original'] = mapping[shuffled_pos]
    
    results.append(result_row)

final_df = pd.DataFrame(results)
final_df.to_csv(data_dir / 'responses_mapped_to_original.tsv', sep="\t", index=False)
print(f"\n✓ Mapped {len(final_df)} rows to original positions")

# ========== ACCURACY CALCULATION ==========
responder_cols = [f'r{i}_original' for i in range(1, 10)]

# Calculate per-responder accuracy
print("\n" + "="*50)
print("ACCURACY METRICS")
print("="*50)

accuracies = {}
for col in responder_cols:
    correct = (final_df[col] == 1).sum()
    total = final_df[col].notna().sum()
    accuracy = (correct / total * 100) if total > 0 else 0
    accuracies[col] = accuracy
    print(f"{col}: {correct}/{total} = {accuracy:.2f}%")

# Overall accuracy
all_responses = final_df[responder_cols].values.flatten()
all_responses = all_responses[~pd.isna(all_responses)]
overall_correct = (all_responses == 1).sum()
overall_total = len(all_responses)
overall_accuracy = (overall_correct / overall_total * 100) if overall_total > 0 else 0

print(f"\n{'OVERALL'}: {overall_correct}/{overall_total} = {overall_accuracy:.2f}%")
# ========== CONFUSION MATRIX ==========
print("\n" + "="*50)
print("CONFUSION MATRIX")
print("="*50)

# Prepare data: true label is always 1, predicted is the response
y_true = []
y_pred = []

for col in responder_cols:
    for response in final_df[col].dropna():
        y_true.append(1)  # Correct answer is always 1
        y_pred.append(int(response))

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4])

print("\nConfusion Matrix (rows=true, cols=predicted):")
print("     Pred1  Pred2  Pred3  Pred4")
for i, row in enumerate(cm, start=1):
    print(f"True{i}:  {row[0]:4d}   {row[1]:4d}   {row[2]:4d}   {row[3]:4d}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[1, 2, 3, 4], 
            yticklabels=[1, 2, 3, 4])
plt.xlabel('Predicted Option')
plt.ylabel('True Option (always 1)')
plt.title('Confusion Matrix: Human Responses')
plt.tight_layout()
plt.savefig(data_dir / 'confusion_matrix.png', dpi=300)
print(f"\n✓ Confusion matrix saved to {data_dir / 'confusion_matrix.png'}")

print(f"\n{final_df.head()}")


