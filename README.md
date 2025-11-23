# AI Ethics Assignment: Designing Responsible and Fair AI Systems

## Overview
This repository contains our comprehensive analysis of AI ethics principles, case studies on algorithmic bias, practical fairness audits, and policy recommendations for responsible AI deployment.

## Deliverables

### Part 1: Conceptual Foundations (30%)
- **File**: DELIVERABLES/PART_1_Conceptual_Foundations.pdf
- **Topics**: Transparency vs. Explainability, GDPR impact, Ethical Principles
- **Grade**: 30% of total

### Part 2: Case Study Analysis (40%)
- **File**: DELIVERABLES/PART_2_Case_Study_Analysis.pdf
- **Cases**: Amazon Hiring Tool, Facial Recognition in Policing
- **Grade**: 40% of total

### Part 3: Practical Audit (25%)
- **Code**: code/compas_bias_audit.py
- **Report**: results/bias_analysis_report.md
- **Visualizations**: results/visualizations/
- **Analysis**: COMPAS recidivism dataset fairness audit
- **Grade**: 25% of total

### Part 4: Ethical Reflection (5%)
- **File**: DELIVERABLES/PART_4_Ethical_Reflection.pdf
- **Prompt**: Personal project ethical AI commitment
- **Grade**: 5% of total

### Bonus: Healthcare Policy (10%)
- **File**: policy/BONUS_Healthcare_AI_Policy.md
- **Content**: 1-page ethical AI guideline for healthcare
- **Grade**: Bonus 10%

## Running the Analysis

### Prerequisites
```bash
python --version  # 3.8+
pip install -r code/requirements.txt
```

### Execute Audit
```bash
cd code/
python compas_bias_audit.py
# Outputs: results/visualizations/ and results/audit_results.json
```

### View Results
```bash
# Check generated visualizations
ls results/visualizations/

# Read audit report
cat results/bias_analysis_report.md
```

## Key Findings (Summary)

### COMPAS Bias Audit
- **False Positive Rate (Black defendants)**: 45%
- **False Positive Rate (White defendants)**: 23%
- **Disparate Impact Ratio**: 1.96x
- **Recommendation**: System requires fairness constraints before deployment

### Policy Recommendations
- Restrict facial recognition use to investigative leads (not arrest basis)
- Require ≥99% accuracy across all demographic groups
- Implement mandatory human review for all high-risk classifications
- Establish 2-year sunset clause requiring re-authorization

## Peer Review Process

All group members reviewed each other's sections:
- Reviews conducted: [Date]
- Feedback consolidated in: PEER_REVIEW/feedback_summary.md
- Changes incorporated: [List of updates]

## Repository Guidelines

- **Branch Structure**: Main branch protected; all changes via pull requests
- **Commit Messages**: Descriptive, reference assignment part (e.g., "PART 2: Case study analysis")
- **Code Style**: PEP 8 compliant; use black formatter
- **Documentation**: Inline comments, docstrings for all functions


## Useful Resources
- [AI Fairness 360 GitHub](https://github.com/Trusted-AI/AIF360)
- [EU Ethics Guidelines](https://ec.europa.eu/digital-single-market/en/news/ethics-guidelines-trustworthy-ai)
- [ProPublica COMPAS Analysis](https://www.propublica.org/article/machine-bias-there-is-software-used-across-the-country-to-predict-future-criminals-and-it-is-biased-against-blacks)

## License
MIT License - See LICENSE file

---

**Last Updated**: 23rd November, 2025