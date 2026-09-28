# SAFE-Judge

**Selective Assessment with a Formal Error guarantee**

This repository contains the codebase for the project **"Calibrated and Selective LLM Relevance Assessment under Domain Shift"**.

## Overview

Building a test collection for Information Retrieval requires expensive human relevance judgments. While Large Language Models (LLMs) can produce these judgments cheaply, completely replacing human assessors is risky due to LLMs being poorly calibrated and sensitive to prompts. 

**SAFE-Judge** is a selective assessment method that:
1. Uses a compact LLM assessor (e.g., a 3B-parameter model) to predict relevance labels and confidence scores.
2. Auto-labels query-document pairs it is confident about.
3. Escalates the remaining uncertain pairs to a human assessor.

By applying **Conformal Risk Control**, SAFE-Judge guarantees a formal upper bound on the expected error rate of its automatic labels (auto-error rate). It ensures a significant reduction in human annotation effort while preserving the ranking of retrieval systems, even when moving from web search (TREC DL) to biomedical search (TREC-COVID).

## Key Features

- **Query-Level Calibration:** Treats queries (not independent pairs) as the calibration unit to accurately reflect how new test collections are built.
- **Separate Class Thresholds:** Employs distinct confidence thresholds for relevant and non-relevant predictions, ensuring error rates are bounded within each predicted class.
- **Domain Shift Evaluation:** Tests the transferability of the error guarantee under a domain shift from web search to a specialised biomedical domain.
