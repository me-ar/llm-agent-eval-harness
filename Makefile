.PHONY: install test lint eval eval-claude smoke
install:
	pip install -r requirements-dev.txt && pip install -e .
test:
	pytest -q
lint:
	ruff check src tests
eval:
	python -m agentkit.harness
eval-claude:
	python -m agentkit.harness --backend anthropic --out reports/claude
smoke:
	python -m agentkit.harness --smoke
