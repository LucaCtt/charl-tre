Goal: Improve deterministic rule-based HAR classifier beyond the current best while keeping interpretability.

Current baseline (saved):
- Accuracy: 0.6111111111111112
- Segments: 216
- Config:
  - classifier_stride=75
  - segment_length=64
  - segment_hop=12
  - top_rules_per_activity=50
  - rule_min_delta=0.0
  - filter_threshold=0.1
  - path_threshold=0.0
  - classifier_train_ratio=0.7

Artifacts to reuse (do NOT rerun LPCMCI unless explicitly needed):
- Raw graphs manifest: out/a1_w75_tw450_b512/causal_graphs_raw/parcorr/manifest.json
- Rule engine outputs: out/a1_w75_tw450_b512/causal_path_search/parcorr/
  - classification_rules.json
  - deterministic_classifier_metrics.json
  - graphs_summary.json

Primary task:
1. Keep the causal-discovery stage decoupled.
2. Optimize only path/rule search and deterministic classifier logic.
3. Target better than 0.6111 accuracy with at least similar sample support (avoid trivial tiny-sample gains).
4. Preserve symbolic rules in the form:
   - IF X[t-lag] is true THEN Y[t] likely true/false
   with sign-aware mapping for true/false.

Suggested directions:
- Activity-specific rule budgets (not a fixed top-k for all classes).
- Rule conflict handling and redundancy pruning.
- Calibration of class priors and confidence margins.
- Better fallback for low-edge activities (e.g., Empty) without collapsing overall accuracy.
- Hierarchical rule scoring (core rules + tie-breaker rules).
- Evaluate Pareto front: accuracy vs. number of segments.

Success criteria:
- New best metrics saved to best_run_summary.json (update or write best_run_summary_v2.json)
- Updated rules exported to classification_rules.json / classification_rules.txt
- Clear comparison table vs baseline.
