import numpy as np
import matplotlib.pyplot as plt

def plot_human_confusion_matrix_matching_style():
    """Create human evaluation plot matching the style of model plots"""
    
    # Your human data
    human_accuracy = 89.58
    confusion_matrix = np.array([
        [387, 16, 23, 6],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
    
    # Use the same labels as your model plots
    display_labels = [
        'Correct\n(Metaphorical)',
        'Distractor 1\n(Concrete)',
        'Distractor 2\n(Abstract)',
        'Distractor 3\n(Antonym)'
    ]
    
    # Create bar plot matching your model style
    fig, ax = plt.subplots(figsize=(12, 6))
    
    values = confusion_matrix[0, :]  # First row since all true are exp1
    x_pos = np.arange(len(display_labels))
    colors = ['#2ecc71', '#e74c3c', '#e67e22', '#c0392b']  # Same colors as models
    
    bars = ax.bar(x_pos, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
               f'{int(val)}\n({val/sum(values)*100:.1f}%)',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Human Predictions', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Predictions', fontsize=14, fontweight='bold')
    ax.set_title(f'Human Performance on Metaphor Detection\nPrompt V1 (Accuracy: {human_accuracy:.1f}%)',
                fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(display_labels, fontsize=12)
    
    # Add explanatory text matching your model plots
    fig.text(0.5, 0.02, 
            'Dataset contains only correct metaphorical instances; Distractors 1-3 are systematic alternatives',
            ha='center', fontsize=10, style='italic', wrap=True)
    
    plt.tight_layout()
    plt.savefig('plots/distractor_analysis_human_v1.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Human confusion matrix saved with matching style")

# Run this
plot_human_confusion_matrix_matching_style()
