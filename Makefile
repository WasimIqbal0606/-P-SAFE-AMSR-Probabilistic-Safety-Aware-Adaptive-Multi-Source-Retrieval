PYTHON ?= .venv/Scripts/python.exe

.PHONY: verify-paper audit-paper test-paper

test-paper:
	$(PYTHON) -m pytest -q

verify-paper:
	$(PYTHON) experiments/run_comprehensive_evidence.py
	$(PYTHON) generate_paper_tables.py
	$(PYTHON) generate_figures.py

audit-paper:
	$(PYTHON) src/psafe/audit_submission.py
