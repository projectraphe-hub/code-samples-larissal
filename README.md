# Code Samples

Selected code samples from two projects demonstrating research methodology and systems thinking.

---

## BIOS-MVP: Bayesian Health Tracking

A health tracking iOS app with a Bayesian correlation engine for detecting relationships between interventions (supplements, behaviors) and health outcomes.

### `bayesian_engine/correlation_detector.py`

Spurious correlation detection system that checks for:
- Temporal trends and autocorrelation
- Day-of-week confounding
- Confounding by other interventions
- Simpson's paradox
- Insufficient variation

Built because health apps love to surface correlations without checking if they're real. This module catches false patterns before users see them.

---

## Project Vess: Persistent Self-Models in LLMs

An empirical investigation of how LLMs behave when given persistent self-models across multiple sessions.

### `analysis/validation_analysis.py`

Semantic diversity analysis using sentence embeddings to validate behavioral metrics. Uses `sentence-transformers` to calculate conceptual spread across model responses.

### `analysis/recode_autonomy.py`

Behavioral coding framework for analyzing LLM responses to researcher pushback. Implements argumentation quality scoring based on structured criteria.

---

Full project write-ups available upon request.
