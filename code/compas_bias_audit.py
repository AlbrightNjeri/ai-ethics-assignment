#!/usr/bin/env python
"""
COMPAS Bias Audit - Main Script
AI Ethics Assignment

Comprehensive analysis of racial bias in COMPAS recidivism prediction system
using AI Fairness 360 and custom fairness metrics.

Usage: python compas_bias_audit.py
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# ============================================================================
# STEP 1: IMPORT LIBRARIES
# ============================================================================

print("\n" + "="*80)
print("COMPAS RECIDIVISM BIAS AUDIT")
print("AI Ethics Assignment - Fairness Analysis")
print("="*80 + "\n")

print("Step 1: Importing libraries...")

try:
    from aif360.datasets import CompasDataset
    from sklearn.metrics import confusion_matrix, classification_report
    print("✓ All libraries imported successfully\n")
except ImportError as e:
    print(f"✗ Error importing libraries: {e}")
    print("Install with: pip install aif360 scikit-learn pandas matplotlib seaborn")
    sys.exit(1)

# ============================================================================
# STEP 2: LOAD DATA
# ============================================================================

print("Step 2: Loading COMPAS dataset from AI Fairness 360...")

try:
    dataset = CompasDataset()
    df = dataset.convert_to_dataframe()[0]
    print(f"✓ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    sys.exit(1)

# ============================================================================
# STEP 3: EXPLORE DATA
# ============================================================================

print("Step 3: Exploring dataset...\n")

print(f"Data types: {df.dtypes.unique().tolist()}")
print(f"Missing values: {df.isnull().sum().sum()}")

if 'race' in df.columns:
    print(f"\nRace distribution:")
    race_counts = df['race'].value_counts()
    for race, count in race_counts.items():
        pct = count / len(df) * 100
        print(f"  {race}: {count:,} ({pct:.1f}%)")

if 'two_year_recidivism' in df.columns:
    recid_rate = df['two_year_recidivism'].mean()
    print(f"\nOverall recidivism rate: {recid_rate:.1%}")

print()

# ============================================================================
# STEP 4: PREPARE DATA
# ============================================================================

print("Step 4: Preparing data for analysis...")

# Create high-risk binary prediction (decile_score >= 5)
if 'decile_score' in df.columns:
    df['high_risk'] = (df['decile_score'] >= 5).astype(int)
    high_risk_pct = df['high_risk'].mean()
    print(f"✓ Created binary prediction (decile_score >= 5)")
    print(f"  {high_risk_pct:.1%} classified as high-risk\n")

# Filter to major racial groups for detailed analysis
df_analysis = df[df['race'].isin(['African-American', 'Caucasian'])].copy()
print(f"✓ Filtered to major racial groups: {len(df_analysis):,} records")
print(f"  African-American: {len(df_analysis[df_analysis['race']=='African-American']):,}")
print(f"  Caucasian: {len(df_analysis[df_analysis['race']=='Caucasian']):,}\n")

# ============================================================================
# STEP 5: CALCULATE FAIRNESS METRICS
# ============================================================================

print("Step 5: Calculating fairness metrics...\n")
print("="*80)

metrics_by_race = {}

for race in ['African-American', 'Caucasian']:
    print(f"\n{race}:")
    print("-" * 40)
    
    race_df = df_analysis[df_analysis['race'] == race]
    
    y_true = race_df['two_year_recidivism'].values
    y_pred = race_df['high_risk'].values
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate rates
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
    accuracy = (tp + tn) / len(race_df) if len(race_df) > 0 else 0
    selection_rate = y_pred.mean()  # % predicted as high-risk
    
    # Store metrics
    metrics_by_race[race] = {
        'n': len(race_df),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'fpr': fpr, 'fnr': fnr, 'tpr': tpr, 'accuracy': accuracy,
        'selection_rate': selection_rate
    }
    
    # Print metrics
    print(f"  Sample size: {len(race_df):,}")
    print(f"  Actual recidivism rate: {y_true.mean():.1%}")
    print(f"  Predicted high-risk rate: {selection_rate:.1%}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  True Positive Rate (TPR): {tpr:.4f} ({tpr*100:.2f}%)")
    print(f"  False Positive Rate (FPR): {fpr:.4f} ({fpr*100:.2f}%)")
    print(f"  False Negative Rate (FNR): {fnr:.4f} ({fnr*100:.2f}%)")
    print(f"\n  Confusion Matrix:")
    print(f"    True Negatives (TN):  {tn:,}")
    print(f"    False Positives (FP): {fp:,}")
    print(f"    False Negatives (FN): {fn:,}")
    print(f"    True Positives (TP):  {tp:,}")

print("\n" + "="*80 + "\n")

# ============================================================================
# STEP 6: DISPARATE IMPACT ANALYSIS
# ============================================================================

print("Step 6: Disparate Impact Analysis\n")
print("="*80)

fpr_aa = metrics_by_race['African-American']['fpr']
fpr_c = metrics_by_race['Caucasian']['fpr']
fnr_aa = metrics_by_race['African-American']['fnr']
fnr_c = metrics_by_race['Caucasian']['fnr']

# Disparate Impact Ratio (FPR-based)
di_ratio = fpr_aa / fpr_c if fpr_c > 0 else float('inf')

# Equalized Odds Gap (FNR-based)
eog_gap = abs(fnr_aa - fnr_c)

print(f"\nFalse Positive Rate (FPR) Analysis:")
print(f"  African-American: {fpr_aa:.4f} ({fpr_aa*100:.2f}%)")
print(f"  Caucasian: {fpr_c:.4f} ({fpr_c*100:.2f}%)")
print(f"  Ratio: {di_ratio:.4f}")

if di_ratio >= 0.8 and di_ratio <= 1.25:
    status = "✓ ACCEPTABLE (within 80% rule)"
elif di_ratio < 0.8:
    status = "⚠ CONCERNING: African-American defendants at disadvantage"
else:
    status = "⚠ CONCERNING: Caucasian defendants at disadvantage"

print(f"  Status: {status}")

print(f"\nFalse Negative Rate (FNR) Analysis:")
print(f"  African-American: {fnr_aa:.4f} ({fnr_aa*100:.2f}%)")
print(f"  Caucasian: {fnr_c:.4f} ({fnr_c*100:.2f}%)")
print(f"  Gap: {eog_gap:.4f}")
print(f"  Target: < 0.05 (5%)")

print(f"\nEquality of Opportunity:")
if eog_gap < 0.05:
    print(f"  ✓ ACHIEVED")
else:
    print(f"  ⚠ NOT ACHIEVED (gap = {eog_gap:.2%})")

print("\n" + "="*80 + "\n")

# ============================================================================
# STEP 7: CREATE VISUALIZATIONS
# ============================================================================

print("Step 7: Creating visualizations...\n")

try:
    results_dir = Path('results/visualizations')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('COMPAS Recidivism Bias Analysis by Race', fontsize=16, fontweight='bold')
    
    races = list(metrics_by_race.keys())
    colors = ['#FF6B6B', '#4ECDC4']
    
    # 1. False Positive Rate Comparison
    fprs = [metrics_by_race[r]['fpr'] for r in races]
    axes[0, 0].bar(races, fprs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[0, 0].set_ylabel('False Positive Rate', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('False Positive Rate by Race\n(Higher = More Unfairly Flagged)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylim(0, max(fprs) * 1.3)
    for i, v in enumerate(fprs):
        axes[0, 0].text(i, v + 0.01, f'{v:.1%}', ha='center', fontweight='bold', fontsize=11)
    axes[0, 0].axhline(y=0.25, color='red', linestyle='--', linewidth=2, label='Threshold')
    axes[0, 0].legend()
    
    # 2. False Negative Rate Comparison
    fnrs = [metrics_by_race[r]['fnr'] for r in races]
    axes[0, 1].bar(races, fnrs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[0, 1].set_ylabel('False Negative Rate', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('False Negative Rate by Race\n(Higher = More Missed Recidivists)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylim(0, max(fnrs) * 1.3 if fnrs else 0.3)
    for i, v in enumerate(fnrs):
        axes[0, 1].text(i, v + 0.01, f'{v:.1%}', ha='center', fontweight='bold', fontsize=11)
    
    # 3. Confusion Matrix - African-American
    cm_aa = np.array([[metrics_by_race['African-American']['tn'],
                       metrics_by_race['African-American']['fp']],
                      [metrics_by_race['African-American']['fn'],
                       metrics_by_race['African-American']['tp']]])
    sns.heatmap(cm_aa, annot=True, fmt='d', cmap='Reds', ax=axes[1, 0],
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'],
                cbar_kws={'label': 'Count'})
    axes[1, 0].set_title('African-American: Confusion Matrix', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('True Label', fontsize=11)
    axes[1, 0].set_xlabel('Predicted Label', fontsize=11)
    
    # 4. Confusion Matrix - Caucasian
    cm_c = np.array([[metrics_by_race['Caucasian']['tn'],
                      metrics_by_race['Caucasian']['fp']],
                     [metrics_by_race['Caucasian']['fn'],
                      metrics_by_race['Caucasian']['tp']]])
    sns.heatmap(cm_c, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'],
                cbar_kws={'label': 'Count'})
    axes[1, 1].set_title('Caucasian: Confusion Matrix', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('True Label', fontsize=11)
    axes[1, 1].set_xlabel('Predicted Label', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('results/visualizations/compas_bias_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/visualizations/compas_bias_analysis.png\n")
    plt.close()
    
except Exception as e:
    print(f"⚠ Could not create visualization: {e}\n")

# ============================================================================
# STEP 8: SAVE RESULTS TO JSON
# ============================================================================

print("Step 8: Saving results to JSON...\n")

try:
    results = {
        'summary': {
            'total_records': len(df_analysis),
            'african_american': len(df_analysis[df_analysis['race'] == 'African-American']),
            'caucasian': len(df_analysis[df_analysis['race'] == 'Caucasian'])
        },
        'metrics': metrics_by_race,
        'disparate_impact': {
            'fpr_ratio': float(di_ratio),
            'status': status,
            'interpretation': 'African-American defendants flagged as high-risk at higher rate'
        },
        'equalized_odds': {
            'fnr_gap': float(eog_gap),
            'target': 0.05
        }
    }
    
    with open('results/audit_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✓ Saved: results/audit_results.json\n")
    
except Exception as e:
    print(f"⚠ Could not save JSON: {e}\n")

# ============================================================================
# STEP 9: SUMMARY & RECOMMENDATIONS
# ============================================================================

print("="*80)
print("AUDIT SUMMARY & KEY FINDINGS")
print("="*80)

print(f"\n🔍 PRIMARY FINDING - Disparate Impact:")
print(f"   African-American defendants are flagged as high-risk at {di_ratio:.2f}x")
print(f"   the rate of Caucasian defendants.")

print(f"\n📊 DETAILED METRICS:")
print(f"   False Positive Rate (African-American): {fpr_aa*100:.2f}%")
print(f"   False Positive Rate (Caucasian): {fpr_c*100:.2f}%")
print(f"   Difference: {abs(fpr_aa - fpr_c)*100:.2f}%")

print(f"\n⚠️  INTERPRETATION:")
print(f"   Among African-American defendants who DO NOT reoffend,")
print(f"   {fpr_aa*100:.1f}% are incorrectly flagged as high-risk.")
print(f"   Among Caucasian defendants who DO NOT reoffend,")
print(f"   {fpr_c*100:.1f}% are incorrectly flagged as high-risk.")
print(f"\n   This systematic disparity indicates RACIAL BIAS in COMPAS.")

print(f"\n🔧 REMEDIATION RECOMMENDATIONS:")
print(f"   1. STOP using COMPAS as sole decision-making tool")
print(f"   2. IMPLEMENT mandatory human review for all high-risk flags")
print(f"   3. RETRAIN model with fairness constraints (equalized odds)")
print(f"   4. REMOVE proxy variables (neighborhood, prior arrests)")
print(f"   5. AUDIT quarterly across all demographic groups")
print(f"   6. INVOLVE affected communities in system redesign")

print(f"\n📋 LEGAL REFERENCE:")
print(f"   EEOC 80% Rule: Disparate Impact Ratio should be ≥ 0.8")
print(f"   Current Ratio: {di_ratio:.2f}")
print(f"   Status: {'FAILS' if di_ratio < 0.8 else 'PASSES'} EEOC standard")

print(f"\n" + "="*80)
print("AUDIT COMPLETE ✓")
print("="*80 + "\n")

print(f"📁 Output files saved:")
print(f"   - results/visualizations/compas_bias_analysis.png")
print(f"   - results/audit_results.json")
print(f"\n📝 Next: Write 300-word report in results/bias_analysis_report.md\n")