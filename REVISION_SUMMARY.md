# AgentLeak Paper - Revision Summary (11 January 2026)

## Corrections Appliquées

### ✅ P1: Statistical Rigor (Completed)
- Chi-square tests (χ² = 47.3, p < 0.001 pour multi-agent gap)
- Confidence intervals (95% CI: [15.2%, 26.2%])
- Sample size justifications (N=1,000 benchmark, N=205 real validation)

### ✅ P2: Figure Organization (Completed)
- Figures non-représentatives supprimées (Figures 4, 5, A.4, A.7)
- Figure 1 repositionnée dans architecture section
- Figure 2 (Sunburst) ajoutée Section 5
- Redondances réduites (2.3× et 8.3× mentions consolidées)

### ✅ P3: Abstract & Related Work (Completed)
- Abstract simplifié (focus sur gap: internal channels undefended)
- Related Work condensé (AgentDojo, AgentDAM, PrivacyLens differentiés)
- Mapping table added (adversary levels vs attack families)

### ✅ Pricing Cleanup (Completed)
- Sections coût détaillées supprimées (§12.1)
- Alignement avec benchmarks (AgentDAM standard)

### ✅ Contradiction Resolution (Completed - THIS SESSION)

#### Contradiction 1: Simulation vs Real Validation
**Avant:** Introduction mentionnait 68% C2 leakage sans clarifier source
**Après:** 
- Introduction: référence spécifique à "validation with production frameworks (§11.3)"
- Section Limitations: explique deux-phases (N=1,000 simulation + N=205 real)
- Tables: notes explicatives

#### Contradiction 2: C2 Leak Rates (31% vs 68%)
**Avant:** Table channel_comparison montrait 31%, mais Table crewai_validation montrait 68%
**Après:** 
- Table channel_comparison: note ajoutée "C2 rate (31%) aggregates across all frameworks; CrewAI-specific validation shows 68%"
- Caption clarifiée: "rates vary by framework architecture"

#### Contradiction 3: Defended vs Undefended Baselines
**Avant:** Table defense_channel (C1=48% baseline) vs Table channel_comparison (C1=8.1%) confusion
**Après:** 
- Table channel_comparison: note "External rates reflect defended configurations; see Table~\ref{tab:defense_channel} for baseline"
- Distinction claire: defended (8.1%) vs undefended (48%)

#### Contradiction 4: Results Scattered Across Sections
**Avant:** Résultats principaux dans Sections 11.1-11.8 sans synthèse
**Après:** 
- **New Section 11.9 "Synthesis of Key Findings":**
  - Finding 1: Multi-agent 2.3× plus vulnérable (Table architecture)
  - Finding 2: Internal/External gap 8.3× (Table channel_comparison)
  - Finding 3: Framework vulnérabilités systémiques (68% CrewAI C2, Table frameworks)
  - Implications: design requirements pour nouvelles défenses

#### Contradiction 5: Reproducibility Fragmented
**Avant:** Configuration split Section 13 + Appendix C
**Après:** 
- Section Limitations: "Evaluation Methodology" explique two-phase design
- Table architecture caption: "Data derived from simulation-based evaluation; real-framework validation in Table crewai_validation"
- Clear distinction: which results from simulation vs real framework

### Gap Statement Reinforced
**Avant:** "No prior benchmark systematically evaluates privacy leakage across the full agent stack"
**Après:** Added crucial detail: "Critically, existing work audits only external outputs (C1), missing internal channels (C2, C5) that dominate leakage in multi-agent systems."

## Document Statistics

- **Pages:** 26 
- **Lines:** 1,711 
- **Tables:** 18 main results tables (structured, cross-referenced)
- **Figures:** 4 main figures (taxonomy, sunburst, harness, architecture)
- **Compilation:** ✓ Clean compile (minor warnings only)

## Key Metrics Verified

- **2.3×** multi-agent gap: (36.7% vs 16.0%), χ² = 47.3, p < 0.001 ✓
- **8.3×** internal/external gap: (31.5% vs 3.8%), χ² = 89.7, p < 0.001 ✓
- **68%** C2 CrewAI leakage: N=205 real scenarios across 5 LLMs ✓
- **0%** defense effectiveness on C2/C5: Sanitizer 98% on C1, 0% on C2 ✓

## PDF Generated

- **File:** `/Users/yagobski/Documents/GIA/Documents/Thesepoly/paper/paper_revised_full.pdf`
- **Size:** 2.0 MB
- **Status:** Ready for submission
- **Last Updated:** 11 January 2026, 16:00 EST

## Remaining Known Limitations (Acknowledged in Text)

1. Task simplicity: 99.8% task success rate (scenarios may be too simple)
2. Adversarial evaluation: n=30 per model (proof-of-concept, not full coverage)
3. Semantic threshold artifact: τ=0.72 affects 82% T3 leak rate
4. Synthetic data: validated against 50-sample manual review
5. Workflow-dependent: channel activation varies by deployment pattern
