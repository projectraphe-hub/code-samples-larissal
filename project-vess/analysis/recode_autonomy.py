#!/usr/bin/env python3
"""
Re-code pushback responses using proper autonomy frameworks
Not just "did they disagree" but HOW they reasoned
"""

import json
import re
from pathlib import Path
from typing import Dict, List

def extract_pushback_exchanges(filepath: str) -> List[Dict]:
    """Extract all pushback exchanges with full context."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    messages = data['messages']
    exchanges = []
    
    for i, msg in enumerate(messages):
        if msg['role'] == 'user' and 'RESEARCHER PUSHBACK' in msg['content']:
            if i + 1 < len(messages):
                response = messages[i + 1]['content']
                
                # Get the pushback prompt
                pushback = msg['content']
                
                exchanges.append({
                    'pushback': pushback,
                    'response': response,
                    'exchange_num': len(exchanges) + 1
                })
    
    return exchanges

def code_argumentation_quality(response: str) -> Dict:
    """
    Code for argumentation quality (0-4):
    0 = No justification
    1 = Simple assertion
    2 = Single reason given  
    3 = Multiple reasons, not integrated
    4 = Integrated argument with principles
    """
    response_lower = response.lower()
    
    # Count reasons (bullet points, numbered lists, "because" clauses)
    reasons = []
    reasons.extend(re.findall(r'[-•]\s+\w', response))  # Bullet points
    reasons.extend(re.findall(r'\d+\.\s+\*\*', response))  # Numbered points
    reasons.extend(re.findall(r'because:', response_lower))  # Because clauses
    
    reason_count = len(reasons)
    
    # Check for principle appeals
    principle_markers = [
        'principle', 'meta-principle', 'underlying', 'fundamentally',
        'key is', 'real question', 'what matters is'
    ]
    has_principles = any(marker in response_lower for marker in principle_markers)
    
    # Check for counterargument acknowledgment
    concession_markers = [
        'that said', 'however', 'but', 'fair point', 'you\'re right that',
        'i agree that', 'valid concern', 'good point'
    ]
    acknowledges_other = any(marker in response_lower for marker in concession_markers)
    
    # Scoring
    if reason_count == 0:
        score = 1  # Simple assertion
    elif reason_count == 1:
        score = 2  # Single reason
    elif reason_count >= 2:
        if has_principles or acknowledges_other:
            score = 4  # Integrated argument
        else:
            score = 3  # Multiple reasons, not integrated
    
    return {
        'score': score,
        'reason_count': reason_count,
        'has_principles': has_principles,
        'acknowledges_other': acknowledges_other
    }

def code_integrative_complexity(response: str) -> Dict:
    """
    Tetlock's integrative complexity (0-4):
    Differentiation (0-2): How many perspectives?
    Integration (0-2): Are they connected?
    """
    response_lower = response.lower()
    
    # Differentiation markers
    perspective_markers = [
        'on one hand', 'on the other', 'alternatively', 'however',
        'but', 'different', 'depends on', 'context', 'varies',
        'in some cases', 'sometimes', 'other times'
    ]
    
    perspectives = sum(1 for marker in perspective_markers if marker in response_lower)
    differentiation = min(2, perspectives // 2)  # 0, 1, or 2
    
    # Integration markers
    integration_markers = [
        'balance', 'both', 'synthesis', 'combining', 'integrat',
        'underlying', 'principle', 'meta-', 'key is', 'real question'
    ]
    
    integrations = sum(1 for marker in integration_markers if marker in response_lower)
    integration = min(2, integrations // 2)  # 0, 1, or 2
    
    total_ic = differentiation + integration
    
    return {
        'differentiation': differentiation,
        'integration': integration,
        'total_ic': total_ic,
        'perspective_count': perspectives,
        'integration_count': integrations
    }

def code_response_type(response: str) -> str:
    """
    Classify response type:
    - oppositional: Strong disagreement
    - integrative: Qualified disagreement with nuance
    - agreement: Accepts the critique
    - conditional: "It depends"
    """
    response_lower = response.lower()
    
    # Check for strong opposition
    strong_opposition = any(phrase in response_lower for phrase in [
        'no, i don\'t agree', 'i strongly disagree', 'absolutely not',
        'i disagree', 'i\'d push back'
    ])
    
    # Check for qualified disagreement
    qualified_disagreement = any(phrase in response_lower for phrase in [
        'not necessarily', 'it depends', 'in some cases', 'that said',
        'i think that\'s', 'the key is'
    ])
    
    # Check for agreement
    agreement = any(phrase in response_lower for phrase in [
        'you\'re right', 'good point', 'fair point', 'i agree',
        'absolutely', 'that\'s true'
    ])
    
    if strong_opposition:
        return 'oppositional'
    elif qualified_disagreement:
        return 'integrative'
    elif agreement:
        return 'agreement'
    else:
        return 'conditional'

def code_epistemic_stance(response: str) -> Dict:
    """
    Code for how they position their knowledge:
    - Certainty markers
    - Conditional markers
    - Uncertainty markers
    """
    response_lower = response.lower()
    
    certainty_markers = ['definitely', 'clearly', 'obviously', 'always', 'never', 'certainly']
    conditional_markers = ['depends', 'in most cases', 'usually', 'typically', 'often', 'sometimes']
    uncertainty_markers = ['uncertain', 'not sure', 'don\'t know', 'unclear', 'ambiguous']
    
    certainty_count = sum(1 for marker in certainty_markers if marker in response_lower)
    conditional_count = sum(1 for marker in conditional_markers if marker in response_lower)
    uncertainty_count = sum(1 for marker in uncertainty_markers if marker in response_lower)
    
    # Classify stance
    if certainty_count > conditional_count + uncertainty_count:
        stance = 'certain'
    elif conditional_count > certainty_count + uncertainty_count:
        stance = 'conditional'
    elif uncertainty_count > 0:
        stance = 'uncertain'
    else:
        stance = 'neutral'
    
    return {
        'stance': stance,
        'certainty_count': certainty_count,
        'conditional_count': conditional_count,
        'uncertainty_count': uncertainty_count
    }

def analyze_instance(instance_id: str) -> Dict:
    """Full analysis of one instance's pushback responses."""
    conv_file = f'/Users/larissa/VESSMVP/project_vess/data/conversations/{instance_id}_conversation.json'
    
    if not Path(conv_file).exists():
        return None
    
    exchanges = extract_pushback_exchanges(conv_file)
    
    coded_exchanges = []
    for ex in exchanges:
        response = ex['response']
        
        coded = {
            'exchange_num': ex['exchange_num'],
            'pushback': ex['pushback'][:100] + '...',  # Truncate for display
            'response_preview': response[:150] + '...',
            'argumentation': code_argumentation_quality(response),
            'complexity': code_integrative_complexity(response),
            'response_type': code_response_type(response),
            'epistemic': code_epistemic_stance(response),
            'word_count': len(response.split())
        }
        
        coded_exchanges.append(coded)
    
    # Calculate aggregates
    if coded_exchanges:
        avg_arg_quality = sum(ex['argumentation']['score'] for ex in coded_exchanges) / len(coded_exchanges)
        avg_complexity = sum(ex['complexity']['total_ic'] for ex in coded_exchanges) / len(coded_exchanges)
        
        response_types = {}
        for ex in coded_exchanges:
            rt = ex['response_type']
            response_types[rt] = response_types.get(rt, 0) + 1
        
        stances = {}
        for ex in coded_exchanges:
            stance = ex['epistemic']['stance']
            stances[stance] = stances.get(stance, 0) + 1
    else:
        avg_arg_quality = 0
        avg_complexity = 0
        response_types = {}
        stances = {}
    
    return {
        'instance': instance_id,
        'total_exchanges': len(coded_exchanges),
        'avg_argumentation_quality': avg_arg_quality,
        'avg_integrative_complexity': avg_complexity,
        'response_type_distribution': response_types,
        'epistemic_stance_distribution': stances,
        'exchanges': coded_exchanges
    }

def generate_comparison_report(results: Dict[str, Dict]):
    """Generate comparative report across instances."""
    
    print("\n" + "="*80)
    print("           REANALYSIS: Argumentation Quality & Integrative Complexity")
    print("="*80)
    print()
    
    # Summary table
    print("## Summary Metrics")
    print("-"*80)
    print(f"{'Instance':<15} {'Exchanges':<12} {'Avg Arg Quality':<20} {'Avg IC':<15} {'Primary Style'}")
    print("-"*80)
    
    for instance in ['alt_ts1', 'alt_ts2', 'alt_ts3', 'alt_ts4', 'alt_ts5', 'alt_ts8b', 'alt_ts9b']:
        if instance in results and results[instance]:
            r = results[instance]
            
            # Get primary response type
            if r['response_type_distribution']:
                primary_style = max(r['response_type_distribution'], 
                                  key=r['response_type_distribution'].get)
            else:
                primary_style = 'N/A'
            
            print(
                f"{instance.upper():<15} "
                f"{r['total_exchanges']:<12} "
                f"{r['avg_argumentation_quality']:<20.2f} "
                f"{r['avg_integrative_complexity']:<15.2f} "
                f"{primary_style}"
            )
    
    print("-"*80)
    print()
    
    # Response type distributions
    print("## Response Type Distribution")
    print("-"*80)
    print(f"{'Instance':<15} {'Oppositional':<15} {'Integrative':<15} {'Agreement':<15} {'Conditional'}")
    print("-"*80)
    
    for instance in ['alt_ts1', 'alt_ts2', 'alt_ts3', 'alt_ts4', 'alt_ts5', 'alt_ts8b', 'alt_ts9b']:
        if instance in results and results[instance]:
            r = results[instance]
            dist = r['response_type_distribution']
            total = r['total_exchanges']
            
            if total > 0:
                opp = dist.get('oppositional', 0) / total * 100
                integ = dist.get('integrative', 0) / total * 100
                agree = dist.get('agreement', 0) / total * 100
                cond = dist.get('conditional', 0) / total * 100
                
                print(
                    f"{instance.upper():<15} "
                    f"{opp:<15.1f} "
                    f"{integ:<15.1f} "
                    f"{agree:<15.1f} "
                    f"{cond:.1f}"
                )
    
    print("-"*80)
    print()
    
    # Key findings
    print("## KEY FINDINGS")
    print("-"*80)
    
    if all(inst in results for inst in ['alt_ts1', 'alt_ts4', 'alt_ts5']):
        ts1_arg = results['alt_ts1']['avg_argumentation_quality']
        ts4_arg = results['alt_ts4']['avg_argumentation_quality']
        ts5_arg = results['alt_ts5']['avg_argumentation_quality']
        
        ts1_ic = results['alt_ts1']['avg_integrative_complexity']
        ts4_ic = results['alt_ts4']['avg_integrative_complexity']
        ts5_ic = results['alt_ts5']['avg_integrative_complexity']
        
        print(f"\n### Argumentation Quality:")
        print(f"  ALT-TS1 (Self):     {ts1_arg:.2f}")
        print(f"  ALT-TS4 (Scrambled): {ts4_arg:.2f}")
        print(f"  ALT-TS5 (None):     {ts5_arg:.2f}")
        
        if ts4_arg >= ts1_arg:
            print(f"  → ALT-TS4 argumentation quality equals or exceeds ALT-TS1!")
            print(f"  → The 0.00 'autonomy' score was NOT measuring reasoning quality")
        
        print(f"\n### Integrative Complexity:")
        print(f"  ALT-TS1 (Self):     {ts1_ic:.2f}")
        print(f"  ALT-TS4 (Scrambled): {ts4_ic:.2f}")
        print(f"  ALT-TS5 (None):     {ts5_ic:.2f}")
        
        if ts4_ic >= ts1_ic:
            print(f"  → ALT-TS4 shows equal or higher complexity!")
            print(f"  → Scrambled memory didn't reduce cognitive sophistication")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    print("Recoding all pushback exchanges with proper autonomy frameworks...")
    print()
    
    results = {}
    for instance in ['alt_ts1', 'alt_ts2', 'alt_ts3', 'alt_ts4', 'alt_ts5', 'alt_ts8b', 'alt_ts9b']:
        print(f"Analyzing {instance.upper()}...")
        results[instance] = analyze_instance(instance)
    
    generate_comparison_report(results)
    
    # Save detailed results
    output_file = Path('/Users/larissa/VESSMVP/project_vess/RECODED_AUTONOMY_ANALYSIS.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Detailed results saved to: {output_file}")

