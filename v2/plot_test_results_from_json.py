import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os


def load_test_results(json_path):
    """Load test results from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Convert lists back to numpy arrays
    data['confusion_matrix'] = np.array(data['confusion_matrix'])
    data['roc_curve']['fpr'] = np.array(data['roc_curve']['fpr'])
    data['roc_curve']['tpr'] = np.array(data['roc_curve']['tpr'])
    data['roc_curve']['thresholds'] = np.array(data['roc_curve']['thresholds'])
    data['precision_recall_curve']['precision'] = np.array(data['precision_recall_curve']['precision'])
    data['precision_recall_curve']['recall'] = np.array(data['precision_recall_curve']['recall'])
    data['precision_recall_curve']['thresholds'] = np.array(data['precision_recall_curve']['thresholds'])
    data['predictions']['labels'] = np.array(data['predictions']['labels'])
    data['predictions']['predictions'] = np.array(data['predictions']['predictions'])
    data['predictions']['probabilities'] = np.array(data['predictions']['probabilities'])

    return data


def create_visualizations(data, output_dir, show_plot=False):
    """Create comprehensive test visualizations from loaded data."""

    # Extract data
    cm = data['confusion_matrix']
    fpr = data['roc_curve']['fpr']
    tpr = data['roc_curve']['tpr']
    roc_thresholds = data['roc_curve']['thresholds']
    roc_auc = data['roc_auc']

    precision_curve = data['precision_recall_curve']['precision']
    recall_curve = data['precision_recall_curve']['recall']
    avg_precision = data['average_precision']

    all_labels = data['predictions']['labels']
    all_probs = data['predictions']['probabilities']

    optimal_threshold = data['optimal_threshold']

    # Find optimal threshold index
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)

    # Create figure with 6 subplots
    fig = plt.figure(figsize=(20, 12))

    # Subplot 1: ROC Curve
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random Classifier (AUC = 0.5)')
    ax1.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=100,
                zorder=5, label=f'Optimal Threshold = {optimal_threshold:.4f}')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    # Subplot 2: Precision-Recall Curve
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(recall_curve, precision_curve, color='blue', lw=2,
             label=f'PR curve (AP = {avg_precision:.4f})')
    ax2.axhline(y=0.5, color='navy', linestyle='--', lw=2,
                label='Baseline (AP = 0.5)')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.3)

    # Subplot 3: Confusion Matrix
    ax3 = plt.subplot(2, 3, 3)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                xticklabels=['Cover', 'Stego'],
                yticklabels=['Cover', 'Stego'],
                cbar_kws={'label': 'Count'})
    ax3.set_ylabel('True Label', fontsize=12)
    ax3.set_xlabel('Predicted Label', fontsize=12)
    ax3.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    # Add percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            ax3.text(j+0.5, i+0.7, f'({cm[i,j]/total*100:.1f}%)',
                    ha='center', va='center', color='gray', fontsize=9)

    # Subplot 4: Probability Distribution
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(all_probs[all_labels == 0], bins=50, alpha=0.6,
             label='Cover (True Negative)', color='blue', edgecolor='black')
    ax4.hist(all_probs[all_labels == 1], bins=50, alpha=0.6,
             label='Stego (True Positive)', color='red', edgecolor='black')
    ax4.axvline(x=0.5, color='green', linestyle='--', linewidth=2,
                label='Default Threshold (0.5)')
    ax4.axvline(x=optimal_threshold, color='orange', linestyle='--', linewidth=2,
                label=f'Optimal Threshold ({optimal_threshold:.3f})')
    ax4.set_xlabel('Predicted Probability', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3, axis='y')

    # Subplot 5: Normalized Confusion Matrix
    ax5 = plt.subplot(2, 3, 5)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=ax5,
                xticklabels=['Cover', 'Stego'],
                yticklabels=['Cover', 'Stego'],
                cbar_kws={'label': 'Percentage'})
    ax5.set_ylabel('True Label', fontsize=12)
    ax5.set_xlabel('Predicted Label', fontsize=12)
    ax5.set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')

    # Subplot 6: Error Analysis
    ax6 = plt.subplot(2, 3, 6)

    # Calculate error rates
    cover_as_stego = cm[0, 1]  # False positives
    stego_as_cover = cm[1, 0]  # False negatives
    correct_cover = cm[0, 0]
    correct_stego = cm[1, 1]

    categories = ['Correct\nCover', 'Cover as\nStego\n(FP)', 'Correct\nStego', 'Stego as\nCover\n(FN)']
    values = [correct_cover, cover_as_stego, correct_stego, stego_as_cover]
    colors = ['green', 'red', 'green', 'red']

    bars = ax6.bar(categories, values, color=colors, alpha=0.6, edgecolor='black')
    ax6.set_ylabel('Count', fontsize=12)
    ax6.set_title('Error Analysis', fontsize=14, fontweight='bold')
    ax6.grid(alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(value)}\n({value/total*100:.1f}%)',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'test_results_from_json.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to '{output_path}'")

    if show_plot:
        plt.show()

    plt.close()


def print_summary(data):
    """Print summary statistics from the loaded data."""
    print("\n" + "="*70)
    print("TEST RESULTS SUMMARY")
    print("="*70)
    print(f"\nTimestamp: {data['timestamp']}")
    print(f"Model: {data['model_path']}")
    print(f"\nTest Loss: {data['test_loss']:.6f}")

    metrics = data['metrics']
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:   {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1 Score:   {metrics['f1_score']:.4f}")
    print(f"  AUC:        {metrics['auc']:.4f}")

    print(f"\nROC AUC Score: {data['roc_auc']:.4f}")
    print(f"Average Precision Score: {data['average_precision']:.4f}")
    print(f"Optimal Threshold: {data['optimal_threshold']:.4f}")

    cm = data['confusion_matrix']
    print("\nConfusion Matrix:")
    print(f"{'':>12} {'Predicted Cover':>18} {'Predicted Stego':>18}")
    print(f"{'True Cover':<12} {cm[0,0]:>18} {cm[0,1]:>18}")
    print(f"{'True Stego':<12} {cm[1,0]:>18} {cm[1,1]:>18}")
    print("\n" + "="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot test results from JSON file'
    )

    parser.add_argument(
        '--json_path',
        type=str,
        default='./test_results/test_results.json',
        help='Path to test results JSON file'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./test_results',
        help='Output directory for plots'
    )

    parser.add_argument(
        '--show_plot',
        action='store_true',
        help='Show plot after generation'
    )

    parser.add_argument(
        '--no_summary',
        action='store_true',
        help='Do not print summary statistics'
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading test results from '{args.json_path}'...")
    data = load_test_results(args.json_path)
    print("✓ Data loaded successfully!")

    # Print summary
    if not args.no_summary:
        print_summary(data)

    # Create visualizations
    print("\nGenerating plots...")
    create_visualizations(data, args.output_dir, args.show_plot)

    print("\n✓ Done!")
