"""
Correlation Detection and Spurious Pattern Warning System
Identifies and warns about potentially spurious correlations
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import stats


class CorrelationDetector:
    """
    Detects spurious correlations and confounding patterns
    
    Checks for:
    - Autocorrelation in outcomes (trend over time)
    - Confounding by day of week
    - Confounding by other interventions
    - Multiple comparison issues
    - Simpson's paradox
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize correlation detector
        
        Args:
            alpha: Significance level for statistical tests
        """
        self.alpha = alpha
    
    def check_for_spurious_correlation(
        self,
        daily_states: List[Dict],
        intervention_name: str,
        outcome_name: str,
        analysis_results: Dict
    ) -> Tuple[bool, List[Dict]]:
        """
        Check if detected correlation might be spurious
        
        Args:
            daily_states: List of daily state dicts
            intervention_name: Name of intervention
            outcome_name: Name of outcome
            analysis_results: Results from Bayesian analysis
            
        Returns:
            (is_likely_spurious, warnings) tuple where:
            - is_likely_spurious: True if correlation is likely spurious
            - warnings: List of warning dicts with type and description
        """
        warnings = []
        
        # Extract time series
        dates = [state['date'] for state in daily_states]
        interventions = [state['interventions'].get(intervention_name, 0) for state in daily_states]
        outcomes = [state['outcomes'].get(outcome_name) for state in daily_states if state['outcomes'].get(outcome_name) is not None]
        
        # 1. Check for temporal trend in outcome
        trend_warning = self._check_temporal_trend(outcomes, dates[:len(outcomes)])
        if trend_warning:
            warnings.append(trend_warning)
        
        # 2. Check for autocorrelation
        autocorr_warning = self._check_autocorrelation(outcomes)
        if autocorr_warning:
            warnings.append(autocorr_warning)
        
        # 3. Check for confounding by day of week
        dow_warning = self._check_day_of_week_confounding(
            daily_states, intervention_name, outcome_name
        )
        if dow_warning:
            warnings.append(dow_warning)
        
        # 4. Check for confounding by other interventions
        other_interv_warnings = self._check_other_intervention_confounding(
            daily_states, intervention_name, outcome_name
        )
        warnings.extend(other_interv_warnings)
        
        # 5. Check for insufficient variation
        variation_warning = self._check_sufficient_variation(interventions, outcomes)
        if variation_warning:
            warnings.append(variation_warning)
        
        # 6. Check for Simpson's paradox
        simpson_warning = self._check_simpsons_paradox(
            daily_states, intervention_name, outcome_name
        )
        if simpson_warning:
            warnings.append(simpson_warning)
        
        # 7. Check effect size vs noise
        noise_warning = self._check_effect_vs_noise(
            analysis_results, outcomes
        )
        if noise_warning:
            warnings.append(noise_warning)
        
        # Determine if likely spurious
        critical_warnings = [w for w in warnings if w['severity'] == 'critical']
        is_likely_spurious = len(critical_warnings) >= 2 or any(
            w['type'] in ['strong_temporal_trend', 'strong_autocorrelation'] 
            for w in critical_warnings
        )
        
        return is_likely_spurious, warnings
    
    def _check_temporal_trend(
        self,
        outcomes: List[float],
        dates: List[str]
    ) -> Optional[Dict]:
        """Check if outcome has strong temporal trend"""
        if len(outcomes) < 10:
            return None
        
        # Linear regression: outcome ~ time
        time_indices = np.arange(len(outcomes))
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            time_indices, outcomes
        )
        
        # Significant trend?
        if p_value < self.alpha and abs(r_value) > 0.5:
            return {
                'type': 'strong_temporal_trend',
                'severity': 'critical',
                'description': (
                    f"Your {self._format_outcome(outcomes)} shows a strong trend over time "
                    f"(r={r_value:.2f}, p={p_value:.4f}). The detected correlation might be "
                    f"due to this overall trend rather than the intervention."
                ),
                'recommendation': (
                    "Continue tracking to see if the trend stabilizes. "
                    "Correlations are more reliable when the outcome is relatively stable."
                ),
                'statistics': {
                    'r_value': float(r_value),
                    'p_value': float(p_value),
                    'slope': float(slope)
                }
            }
        
        return None
    
    def _check_autocorrelation(
        self,
        outcomes: List[float]
    ) -> Optional[Dict]:
        """Check for autocorrelation in outcomes"""
        if len(outcomes) < 15:
            return None
        
        # Lag-1 autocorrelation
        outcomes_arr = np.array(outcomes)
        autocorr = np.corrcoef(outcomes_arr[:-1], outcomes_arr[1:])[0, 1]
        
        if abs(autocorr) > 0.6:
            return {
                'type': 'strong_autocorrelation',
                'severity': 'warning',
                'description': (
                    f"Your {self._format_outcome(outcomes)} from one day strongly predicts "
                    f"the next day (autocorr={autocorr:.2f}). This can inflate correlation estimates."
                ),
                'recommendation': (
                    "The analysis controls for previous-day outcomes, but be aware that "
                    "persistent patterns can still affect results."
                ),
                'statistics': {
                    'lag1_autocorr': float(autocorr)
                }
            }
        
        return None
    
    def _check_day_of_week_confounding(
        self,
        daily_states: List[Dict],
        intervention_name: str,
        outcome_name: str
    ) -> Optional[Dict]:
        """Check if day of week confounds the relationship"""
        # Extract data
        dow = []
        interventions = []
        outcomes = []
        
        for state in daily_states:
            dow_val = state.get('context', {}).get('day_of_week')
            interv_val = state['interventions'].get(intervention_name, 0)
            outcome_val = state['outcomes'].get(outcome_name)
            
            if dow_val is not None and outcome_val is not None:
                dow.append(dow_val)
                interventions.append(interv_val)
                outcomes.append(outcome_val)
        
        if len(dow) < 14:  # Need at least 2 weeks
            return None
        
        # Check if intervention varies with day of week
        # (e.g., only take supplement on weekdays)
        intervention_by_dow = {}
        for d, i in zip(dow, interventions):
            if d not in intervention_by_dow:
                intervention_by_dow[d] = []
            intervention_by_dow[d].append(i)
        
        # Calculate proportion of days with intervention per DOW
        dow_intervention_rates = {
            d: np.mean([1 if x > 0 else 0 for x in vals])
            for d, vals in intervention_by_dow.items()
        }
        
        # If rate varies a lot (e.g., 90% on weekdays, 10% on weekends)
        if dow_intervention_rates:
            rate_std = np.std(list(dow_intervention_rates.values()))
            if rate_std > 0.3:  # High variance in usage by DOW
                # Check if outcome also varies by DOW
                outcome_by_dow = {}
                for d, o in zip(dow, outcomes):
                    if d not in outcome_by_dow:
                        outcome_by_dow[d] = []
                    outcome_by_dow[d].append(o)
                
                dow_outcome_means = {
                    d: np.mean(vals) for d, vals in outcome_by_dow.items()
                }
                
                outcome_dow_var = np.var(list(dow_outcome_means.values()))
                
                if outcome_dow_var > 0.5:  # Outcome also varies by DOW
                    return {
                        'type': 'day_of_week_confounding',
                        'severity': 'warning',
                        'description': (
                            f"You tend to take {intervention_name.replace('_', ' ')} on certain days "
                            f"of the week, and your {self._format_outcome(outcomes)} also varies by day. "
                            f"The analysis controls for this, but be aware of potential confounding."
                        ),
                        'recommendation': (
                            "Try varying when you take this intervention to help isolate its effect."
                        ),
                        'statistics': {
                            'intervention_rate_std': float(rate_std),
                            'outcome_dow_variance': float(outcome_dow_var)
                        }
                    }
        
        return None
    
    def _check_other_intervention_confounding(
        self,
        daily_states: List[Dict],
        intervention_name: str,
        outcome_name: str
    ) -> List[Dict]:
        """Check if other interventions are confounding"""
        warnings = []
        
        # Find other interventions that co-occur frequently
        co_occurrences = {}
        
        for state in daily_states:
            has_target = state['interventions'].get(intervention_name, 0) > 0
            if has_target:
                for other_interv, val in state['interventions'].items():
                    if other_interv != intervention_name and val > 0:
                        if other_interv not in co_occurrences:
                            co_occurrences[other_interv] = 0
                        co_occurrences[other_interv] += 1
        
        # Check for high co-occurrence
        target_count = sum(1 for s in daily_states if s['interventions'].get(intervention_name, 0) > 0)
        
        for other_interv, count in co_occurrences.items():
            co_occurrence_rate = count / target_count if target_count > 0 else 0
            
            if co_occurrence_rate > 0.8:  # Co-occurs 80%+ of the time
                warnings.append({
                    'type': 'intervention_confounding',
                    'severity': 'warning',
                    'description': (
                        f"You often take {other_interv.replace('_', ' ')} on the same days as "
                        f"{intervention_name.replace('_', ' ')} ({co_occurrence_rate*100:.0f}% of the time). "
                        f"The detected effect might be from {other_interv.replace('_', ' ')} instead."
                    ),
                    'recommendation': (
                        f"Try taking {intervention_name.replace('_', ' ')} without "
                        f"{other_interv.replace('_', ' ')} sometimes to isolate effects."
                    ),
                    'statistics': {
                        'co_occurrence_rate': float(co_occurrence_rate),
                        'confounding_intervention': other_interv
                    }
                })
        
        return warnings
    
    def _check_sufficient_variation(
        self,
        interventions: List[float],
        outcomes: List[float]
    ) -> Optional[Dict]:
        """Check if there's sufficient variation in data"""
        # Check outcome variation
        if len(outcomes) < 3:
            return None
        
        outcome_std = np.std(outcomes)
        outcome_range = max(outcomes) - min(outcomes)
        
        if outcome_range < 1.0 and outcome_std < 0.5:
            return {
                'type': 'low_outcome_variation',
                'severity': 'warning',
                'description': (
                    f"Your {self._format_outcome(outcomes)} values are very consistent "
                    f"(range: {outcome_range:.1f}, std: {outcome_std:.2f}). "
                    f"Low variation makes it hard to detect meaningful effects."
                ),
                'recommendation': (
                    "This might indicate stable wellbeing (good!) or a narrow tracking scale. "
                    "Consider using the full range of your scale."
                ),
                'statistics': {
                    'outcome_std': float(outcome_std),
                    'outcome_range': float(outcome_range)
                }
            }
        
        return None
    
    def _check_simpsons_paradox(
        self,
        daily_states: List[Dict],
        intervention_name: str,
        outcome_name: str
    ) -> Optional[Dict]:
        """Check for Simpson's paradox (effect reverses when stratified)"""
        # Stratify by weekend vs weekday
        weekday_states = []
        weekend_states = []
        
        for state in daily_states:
            is_weekend = state.get('context', {}).get('is_weekend', False)
            if is_weekend:
                weekend_states.append(state)
            else:
                weekday_states.append(state)
        
        if len(weekday_states) < 5 or len(weekend_states) < 5:
            return None
        
        # Compute correlation in each group
        def compute_correlation(states):
            interventions = [s['interventions'].get(intervention_name, 0) for s in states]
            outcomes = [s['outcomes'].get(outcome_name) for s in states if s['outcomes'].get(outcome_name) is not None]
            
            if len(interventions) != len(outcomes) or len(outcomes) < 5:
                return None
            
            return stats.pearsonr(interventions[:len(outcomes)], outcomes)[0]
        
        weekday_corr = compute_correlation(weekday_states)
        weekend_corr = compute_correlation(weekend_states)
        
        if weekday_corr is not None and weekend_corr is not None:
            # Check if signs differ
            if np.sign(weekday_corr) != np.sign(weekend_corr) and abs(weekday_corr) > 0.3 and abs(weekend_corr) > 0.3:
                return {
                    'type': 'simpsons_paradox',
                    'severity': 'critical',
                    'description': (
                        f"The relationship between {intervention_name.replace('_', ' ')} and "
                        f"{self._format_outcome([])} appears to reverse on weekdays vs weekends "
                        f"(weekday corr: {weekday_corr:.2f}, weekend corr: {weekend_corr:.2f}). "
                        f"This suggests confounding."
                    ),
                    'recommendation': (
                        "The overall correlation might be misleading. "
                        "Consider weekday/weekend context when interpreting results."
                    ),
                    'statistics': {
                        'weekday_correlation': float(weekday_corr),
                        'weekend_correlation': float(weekend_corr)
                    }
                }
        
        return None
    
    def _check_effect_vs_noise(
        self,
        analysis_results: Dict,
        outcomes: List[float]
    ) -> Optional[Dict]:
        """Check if detected effect is large relative to noise"""
        best_lag = analysis_results['best_lag']
        mean_effect = best_lag['stats']['mean']
        ci_width = best_lag['stats']['ci_upper'] - best_lag['stats']['ci_lower']
        outcome_std = np.std(outcomes)
        
        # If confidence interval is very wide relative to effect
        if ci_width > abs(mean_effect) * 3:
            return {
                'type': 'high_uncertainty',
                'severity': 'warning',
                'description': (
                    f"The detected effect has high uncertainty "
                    f"(95% CI width: {ci_width:.2f} vs effect: {abs(mean_effect):.2f}). "
                    f"The true effect could be much larger or smaller."
                ),
                'recommendation': (
                    "More data will help narrow down the true effect size. "
                    "Continue tracking for a clearer picture."
                ),
                'statistics': {
                    'ci_width': float(ci_width),
                    'effect_mean': float(mean_effect),
                    'uncertainty_ratio': float(ci_width / abs(mean_effect)) if mean_effect != 0 else float('inf')
                }
            }
        
        return None
    
    def _format_outcome(self, outcomes: List[float]) -> str:
        """Format outcome name for display"""
        # This is a placeholder - would be extracted from actual outcome name
        return "wellbeing measure"


# Example usage
if __name__ == "__main__":
    # Simulate data with spurious correlation
    daily_states = [
        {
            'date': f'2025-10-{i:02d}',
            'interventions': {'magnesium': 400 if i % 2 == 0 else 0},
            'outcomes': {'mood_score': 3 + i * 0.1},  # Linear trend
            'context': {'day_of_week': i % 7, 'is_weekend': (i % 7) in [5, 6]}
        }
        for i in range(1, 29)
    ]
    
    analysis_results = {
        'best_lag': {
            'stats': {
                'mean': 0.5,
                'ci_lower': -0.2,
                'ci_upper': 1.2
            }
        }
    }
    
    detector = CorrelationDetector()
    is_spurious, warnings = detector.check_for_spurious_correlation(
        daily_states,
        'magnesium',
        'mood_score',
        analysis_results
    )
    
    print(f"Likely spurious: {is_spurious}")
    print(f"\nWarnings ({len(warnings)}):")
    for warning in warnings:
        print(f"\n{warning['severity'].upper()}: {warning['type']}")
        print(f"  {warning['description']}")
        print(f"  Recommendation: {warning['recommendation']}")


