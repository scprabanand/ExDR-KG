# Convenience wrappers; run from the repository root: make -f rerun/Makefile <target>
DRIVER = bash rerun/scripts/run_pipeline.sh
.PHONY: help test splits graph-baselines main-eval ablation hallucination hkg-stats paths cases rlhf-audit depth backbones llm-baselines facts figures language
help:
	@$(DRIVER) help
test:
	PYTHONPATH=rerun python -m pytest rerun/tests -q
splits graph-baselines main-eval ablation hallucination hkg-stats paths cases rlhf-audit depth backbones llm-baselines facts figures language:
	$(DRIVER) $@
