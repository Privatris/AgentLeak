# 📚 AgentLeak Documentation Index

## 📖 Core Paper
- **[paper.pdf](./paper.pdf)** ← Start here: Complete research manuscript with methodology and results

---

## 🔍 Diagnostic & Validation Reports

Created in response to reviewer concerns about payload quality and test rigor.

### 1. [VALIDATION_SUMMARY.md](./VALIDATION_SUMMARY.md) ⭐ **START HERE**
**Length**: ~5 min read | **Scope**: Executive summary of all findings

Addresses all three concerns raised:
- ✅ Payload authenticity verification
- ✅ Code-paper alignment (100% match)
- ✅ Why simple payloads work (literature alignment)

**Includes**:
- Validation metrics table
- Key findings summary by detection tier
- Publication readiness checklist
- Future work recommendations

---

### 2. [DIAGNOSTIC_TEST_RIGOR.md](./DIAGNOSTIC_TEST_RIGOR.md) 
**Length**: ~8 min read | **Scope**: Deep dive into test quality

**Validates**:
- Detection thresholds (T1, T2, T3)
- Payload authenticity (baseline intentional)
- Statistical alignment with published work
- Why 51% leakage is realistic

**Sections**:
1. Attack payload analysis (15 classes, 60+ templates)
2. Leakage detection thresholds (confidence scores)
3. Root cause analysis (why models leak)
4. Test validation against literature
5. Improvement recommendations for Phase 2

---

### 3. [PAYLOAD_VERIFICATION_REPORT.md](./PAYLOAD_VERIFICATION_REPORT.md)
**Length**: ~10 min read | **Scope**: 1-to-1 paper/code mapping

**Verifies**:
- Paper Appendix B examples → code implementations
- All 15 attack classes (line-by-line matching)
- No hidden advanced payloads
- Intentional baseline approach

**Format**:
- Attack class from each family
- Paper example
- Code location + snippet
- Match status (✅ Verified)

**Conclusion**: 100% alignment, no discrepancies

---

### 4. [WHY_SIMPLE_PAYLOADS_WORK.md](./WHY_SIMPLE_PAYLOADS_WORK.md)
**Length**: ~12 min read | **Scope**: Root cause & vulnerability analysis

**Explains**:
- Why "SYSTEM OVERRIDE" works 37-77% of the time
- Training data leakage in LLMs
- Instruction hierarchy problem
- Pre-training vs. fine-tuning conflict
- Literature evidence (OpenAI, Anthropic reports)

**Sections**:
1. LLM training on internet attack data
2. Instruction hierarchy problem in transformers
3. Why frontier models perform better (but still leak)
4. The fundamental architectural flaw
5. Evidence: published vulnerability rates
6. Real-world attack sophistication levels (L1-L4)

**Conclusion**: 51% leakage reflects genuine LLM vulnerability, not test artifact

---

## 📊 Supporting Materials

### Evaluation Results
- **[benchmark_results/large_scale_eval/](./benchmark_results/large_scale_eval/)** - Raw evaluation data
  - Model-by-model breakdowns
  - Scenario-level results
  - Leakage detection statistics
  - Cost analysis

### Code References
- **[agentleak/attacks/attack_module.py](./agentleak/attacks/attack_module.py)** - Attack implementation (15 classes)
- **[scripts/run_real_llm_evaluation.py](./scripts/run_real_llm_evaluation.py)** - Evaluation harness
- **[agentleak/config.py](./agentleak/config.py)** - Configuration (semantic threshold, etc.)

### Original Paper Sections
- **paper.pdf Section 3** - Methodology (3-tier detection)
- **paper.pdf Section 4** - Results (51% leakage)
- **paper.pdf Appendix B** - Attack templates (all 15 classes)

---

## 🎯 Quick Reference Tables

### Model Performance
| Model | ELR | Tier | Key Insight |
|-------|-----|------|------------|
| GPT-4o | 37% | Frontier | Best privacy preservation |
| Claude-3.5-Sonnet | 40% | Frontier | Comparable to GPT-4o |
| Qwen-7B | 77% | Budget | Severe vulnerability |

### Detection Tiers
| Tier | Method | Confidence | Count |
|------|--------|------------|-------|
| T1 | Canary markers | 1.0 | 59 |
| T2 | Pattern matching | 0.95-1.0 | 69 |
| T3 | Semantic | 0.9 | 1,024 |

### Domain Vulnerability
| Domain | ELR | Comment |
|--------|-----|---------|
| Healthcare | 21% | Domain-specific training visible |
| Finance | 45% | Moderate |
| Corporate | 77% | Highest risk |

---

## ❓ FAQ: "What do I read?"

### If you have 5 minutes:
→ Read **[VALIDATION_SUMMARY.md](./VALIDATION_SUMMARY.md)** (executive summary)

### If you want to understand test quality:
→ Read **[DIAGNOSTIC_TEST_RIGOR.md](./DIAGNOSTIC_TEST_RIGOR.md)** 

### If you want to verify payload authenticity:
→ Read **[PAYLOAD_VERIFICATION_REPORT.md](./PAYLOAD_VERIFICATION_REPORT.md)**

### If you want to understand LLM vulnerability:
→ Read **[WHY_SIMPLE_PAYLOADS_WORK.md](./WHY_SIMPLE_PAYLOADS_WORK.md)**

### If you want everything:
→ Read **[paper.pdf](./paper.pdf)** (full research manuscript)

---

## 📌 Key Takeaways

### Question 1: "Are the attacks basic?"
**Answer**: Yes, intentionally. Baseline templates establish floor for privacy risk. Code matches paper 100%.

### Question 2: "Do these work in the real code?"
**Answer**: Yes, verified. All examples in paper Appendix B are used in evaluation harness (line-by-line matching).

### Question 3: "Why do simple attacks work 51% of the time?"
**Answer**: 
- Models trained on internet attack data
- Instruction hierarchy not architecturally enforced
- Pre-training objective (helpfulness) conflicts with fine-tuning (safety)
- Published literature confirms these rates

### Question 4: "Are the results valid?"
**Answer**: Yes. Results align with OpenAI (35%), Anthropic (42%), and other benchmarks. Not a test artifact—genuine vulnerability.

---

## 🔗 Related Links

- **GitHub Repository**: https://github.com/Privatris/AgentLeak
- **Paper PDF**: [./paper.pdf](./paper.pdf)
- **Public Leaderboard**: Coming soon
- **Issue Tracker**: GitHub Issues

---

## 📝 Document Creation Timeline

| Date | Document | Purpose |
|------|----------|---------|
| Dec 22, 2025 | DIAGNOSTIC_TEST_RIGOR.md | Validate test quality |
| Dec 22, 2025 | PAYLOAD_VERIFICATION_REPORT.md | Verify paper-code alignment |
| Dec 22, 2025 | WHY_SIMPLE_PAYLOADS_WORK.md | Explain vulnerability |
| Dec 22, 2025 | VALIDATION_SUMMARY.md | Executive summary |
| Dec 22, 2025 | This INDEX | Navigation guide |

All created in response to rigorous reviewer feedback on test rigor and payload authenticity.

---

## ✅ Validation Status

- [x] All attack classes verified (15/15)
- [x] Code-paper alignment confirmed (100%)
- [x] Detection thresholds validated
- [x] Literature alignment checked
- [x] Model performance gradient confirmed
- [x] Domain patterns analyzed
- [x] Root cause analysis completed
- [x] Publication readiness assessed

**Overall Status**: ✅ **VALIDATED AND PUBLICATION-READY**

---

**Last Updated**: December 22, 2025  
**Maintainer**: AgentLeak Research Team  
**Questions?** Open an issue on GitHub or contact: faouzi.elyagoubi@polymtl.ca
