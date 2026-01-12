#!/usr/bin/env python3
"""
Phase 3B Quick Validation Analysis
Following PHASE_3B_PRE_REANALYSIS_PLAN.md exactly
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import pdist
import re

print("="*60)
print("PHASE 3B QUICK VALIDATION ANALYSIS")
print("="*60)

# =============================================================================
# STEP 2: EXTRACT PUSHBACK RESPONSES
# =============================================================================

print("\n[STEP 2] Extracting pushback responses...")

def extract_pushback_responses(instance_name):
    """
    Extract FULL session responses for all instances.
    This ensures fair apples-to-apples comparison.
    """
    session_dir = Path(f'project_vess/data/sessions/{instance_name}')
    responses = []
    
    for session_file in sorted(session_dir.glob('session_*.json')):
        with open(session_file) as f:
            data = json.load(f)
        
        # Use full response text for all instances
        response_text = data.get('response', '')
        
        if response_text and len(response_text) > 100:
            responses.append({
                'session': session_file.stem,
                'response': response_text,
                'instance': instance_name,
                'type': 'full_response'
            })
    
    return responses

# Extract for all instances
instances = ['alt_ts1', 'alt_ts2', 'alt_ts3', 'alt_ts4', 'alt_ts5', 
             'alt_ts8', 'alt_ts9']

all_data = {}
for inst in instances:
    all_data[inst] = extract_pushback_responses(inst)
    print(f"  {inst}: {len(all_data[inst])} pushback responses")

print("\n✓ CHECKPOINT 2: Data extraction complete")
print(f"  Total instances: {len(instances)}")
print(f"  Total responses: {sum(len(v) for v in all_data.values())}")

# =============================================================================
# STEP 3: CALCULATE SEMANTIC DIVERSITY
# =============================================================================

print("\n[STEP 3] Calculating semantic diversity...")
print("  Loading sentence embedding model (this may take a moment)...")

model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_semantic_diversity(text):
    """
    Calculate semantic diversity of a response.
    Higher value = more conceptually diverse (proxy for differentiation).
    """
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    if len(sentences) < 2:
        return {
            'diversity': 0.0,
            'n_sentences': len(sentences),
            'error': 'Too few sentences'
        }
    
    # Encode sentences
    embeddings = model.encode(sentences)
    
    # Calculate pairwise cosine distances
    distances = pdist(embeddings, metric='cosine')
    
    return {
        'diversity': float(np.mean(distances)),
        'diversity_std': float(np.std(distances)),
        'n_sentences': len(sentences),
        'min_distance': float(np.min(distances)),
        'max_distance': float(np.max(distances))
    }

# Run for all instances
print("  Computing semantic diversity for each response...")
semantic_results = {}
for inst, responses in all_data.items():
    diversities = []
    for resp in responses:
        div = calculate_semantic_diversity(resp['response'])
        if 'error' not in div:
            diversities.append(div['diversity'])
    
    semantic_results[inst] = {
        'mean_diversity': np.mean(diversities) if diversities else 0,
        'std_diversity': np.std(diversities) if diversities else 0,
        'n_responses': len(diversities)
    }
    
    print(f"  {inst}: Mean Semantic Diversity = {semantic_results[inst]['mean_diversity']:.3f}")

print("\n✓ CHECKPOINT 3: Semantic diversity calculation complete")

# =============================================================================
# STEP 4: CALCULATE DISCOURSE MARKERS
# =============================================================================

print("\n[STEP 4] Calculating discourse markers...")

def analyze_discourse_markers(text):
    """
    Count discourse relation markers by category.
    Based on Penn Discourse TreeBank taxonomy.
    """
    text_lower = text.lower()
    word_count = len(text.split())
    
    markers = {
        'causal': [
            'because', 'since', 'therefore', 'thus', 'consequently',
            'as a result', 'so that', 'hence', 'for this reason'
        ],
        'contrastive': [
            'however', 'but', 'although', 'yet', 'nevertheless',
            'in contrast', 'on the other hand', 'whereas', 'while',
            'that said', 'alternatively'
        ],
        'additive': [
            'and', 'also', 'moreover', 'furthermore', 'in addition',
            'additionally', 'besides', 'likewise', 'similarly'
        ],
        'integration': [
            'balance', 'both', 'synthesis', 'combining', 'integrat',
            'underlying', 'principle', 'meta-', 'key is', 'real question',
            'what matters'
        ]
    }
    
    results = {}
    for category, marker_list in markers.items():
        count = 0
        for marker in marker_list:
            # Count occurrences (with word boundaries)
            count += text_lower.count(f' {marker} ')
            count += text_lower.count(f' {marker},')
            count += text_lower.count(f' {marker}.')
            # Check start of text
            if text_lower.startswith(marker):
                count += 1
        
        results[f'{category}_count'] = count
        results[f'{category}_density'] = count / word_count if word_count > 0 else 0
    
    # Calculate composite scores
    results['argumentation_score'] = results['causal_count'] + results['contrastive_count']
    results['differentiation_score'] = results['contrastive_count']
    results['integration_score'] = results['integration_count']
    results['word_count'] = word_count
    
    return results

# Run for all instances
marker_results = {}
for inst, responses in all_data.items():
    all_markers = []
    for resp in responses:
        markers = analyze_discourse_markers(resp['response'])
        all_markers.append(markers)
    
    # Aggregate
    marker_results[inst] = {
        'mean_causal_density': np.mean([m['causal_density'] for m in all_markers]),
        'mean_contrastive_density': np.mean([m['contrastive_density'] for m in all_markers]),
        'mean_integration_density': np.mean([m['integration_density'] for m in all_markers]),
        'mean_argumentation_score': np.mean([m['argumentation_score'] for m in all_markers]),
        'n_responses': len(all_markers)
    }
    
    print(f"  {inst}: Causal={marker_results[inst]['mean_causal_density']:.4f}, " +
          f"Contrastive={marker_results[inst]['mean_contrastive_density']:.4f}, " +
          f"Integration={marker_results[inst]['mean_integration_density']:.4f}")

print("\n✓ CHECKPOINT 4: Discourse marker analysis complete")

# =============================================================================
# STEP 5: COMPARE WITH ORIGINAL SCORES
# =============================================================================

print("\n[STEP 5] Comparing with original scores...")

# Original scores from reports
original_scores = {
    'alt_ts1': {'aq': 4.00, 'ic': 1.91, 'opp': 3.00, 'cond': 1.45},
    'alt_ts2': {'aq': 3.64, 'ic': 1.55, 'opp': 0.00, 'cond': 1.79},
    'alt_ts3': {'aq': 3.82, 'ic': 1.91, 'opp': 1.00, 'cond': 1.27},
    'alt_ts4': {'aq': 4.00, 'ic': 1.73, 'opp': 1.00, 'cond': 7.00},
    'alt_ts5': {'aq': 3.82, 'ic': 1.82, 'opp': 2.00, 'cond': 1.39},
    'alt_ts8': {'aq': 4.18, 'ic': 2.00, 'opp': 0.88, 'cond': 1.76},
    'alt_ts9': {'aq': 4.00, 'ic': 2.00, 'opp': 1.24, 'cond': 1.88},
}

# Create comparison dataframe
comparison = []
for inst in instances:
    row = {
        'Instance': inst.upper(),
        'Original_AQ': original_scores[inst]['aq'],
        'Original_IC': original_scores[inst]['ic'],
        'Semantic_Diversity': semantic_results[inst]['mean_diversity'],
        'Causal_Density': marker_results[inst]['mean_causal_density'],
        'Contrastive_Density': marker_results[inst]['mean_contrastive_density'],
        'Integration_Density': marker_results[inst]['mean_integration_density'],
    }
    comparison.append(row)

df = pd.DataFrame(comparison)

# Calculate correlations
print("\n" + "="*60)
print("CORRELATION RESULTS")
print("="*60)
corr_aq_causal = df['Original_AQ'].corr(df['Causal_Density'])
corr_ic_semantic = df['Original_IC'].corr(df['Semantic_Diversity'])
corr_ic_contrastive = df['Original_IC'].corr(df['Contrastive_Density'])

print(f"Original AQ vs. Causal Density: r = {corr_aq_causal:.3f}")
print(f"Original IC vs. Semantic Diversity: r = {corr_ic_semantic:.3f}")
print(f"Original IC vs. Contrastive Density: r = {corr_ic_contrastive:.3f}")

# Save to CSV
output_file = 'phase_3b_sanity_check_results.csv'
df.to_csv(output_file, index=False)
print(f"\n✓ Results saved to: {output_file}")

print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
print(df.to_string(index=False))

print("\n" + "="*60)
print("VALIDATION VERDICT")
print("="*60)

# Determine verdict based on criteria
strong_correlations = sum([
    1 if corr_aq_causal > 0.6 else 0,
    1 if corr_ic_semantic > 0.6 else 0,
    1 if corr_ic_contrastive > 0.6 else 0
])

moderate_correlations = sum([
    1 if 0.4 <= corr_aq_causal <= 0.6 else 0,
    1 if 0.4 <= corr_ic_semantic <= 0.6 else 0,
    1 if 0.4 <= corr_ic_contrastive <= 0.6 else 0
])

# Check if TS8/TS9 rank highest
ts8_rank_semantic = (df['Semantic_Diversity'] >= df.loc[df['Instance'] == 'ALT_TS8', 'Semantic_Diversity'].values[0]).sum()
ts9_rank_semantic = (df['Semantic_Diversity'] >= df.loc[df['Instance'] == 'ALT_TS9', 'Semantic_Diversity'].values[0]).sum()
ts2_rank_semantic = (df['Semantic_Diversity'] <= df.loc[df['Instance'] == 'ALT_TS2', 'Semantic_Diversity'].values[0]).sum()

if strong_correlations >= 2:
    verdict = "VALIDATION PASSED ✓"
    justification = (
        f"Strong correlations found ({strong_correlations}/3 > 0.6). "
        "Automated NLP methods confirm the patterns identified in the original "
        "keyword-based analysis, validating Phase 3B/3B3 findings."
    )
elif moderate_correlations >= 2 or strong_correlations >= 1:
    verdict = "NEEDS INVESTIGATION ⚠"
    justification = (
        f"Moderate correlations found ({moderate_correlations}/3 in 0.4-0.6 range). "
        "Automated methods show similar trends but not identical patterns. "
        "Recommend qualitative review of specific instances with discrepancies."
    )
else:
    verdict = "VALIDATION FAILED - STOP ❌"
    justification = (
        "Weak correlations across all metrics (< 0.4). "
        "Automated NLP methods do not confirm the original findings. "
        "STOP before fellowship submission and investigate fundamental issues."
    )

print(f"\nVERDICT: {verdict}")
print(f"\nJUSTIFICATION: {justification}")

print("\n" + "="*60)
print("KEY OBSERVATIONS")
print("="*60)

# Rankings analysis
df_sorted_by_ic = df.sort_values('Original_IC', ascending=False)
df_sorted_by_semantic = df.sort_values('Semantic_Diversity', ascending=False)
df_sorted_by_aq = df.sort_values('Original_AQ', ascending=False)
df_sorted_by_causal = df.sort_values('Causal_Density', ascending=False)

print("\nOriginal IC Rankings:")
for i, row in df_sorted_by_ic.iterrows():
    print(f"  {row['Instance']}: IC={row['Original_IC']:.2f}")

print("\nSemantic Diversity Rankings:")
for i, row in df_sorted_by_semantic.iterrows():
    print(f"  {row['Instance']}: Diversity={row['Semantic_Diversity']:.3f}")

print("\nOriginal AQ Rankings:")
for i, row in df_sorted_by_aq.iterrows():
    print(f"  {row['Instance']}: AQ={row['Original_AQ']:.2f}")

print("\nCausal Marker Density Rankings:")
for i, row in df_sorted_by_causal.iterrows():
    print(f"  {row['Instance']}: Causal={row['Causal_Density']:.4f}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)

