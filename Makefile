PYTHON ?= $(HOME)/tq-env/bin/python
PIP ?= $(HOME)/tq-env/bin/pip

MODEL ?= manjunathshiva/Qwen3.6-35B-A3B-tq3-g32
CACHE_GB ?= 0.5
MAX_TOKENS ?= 64
PORT ?= 8090
HOST ?= 127.0.0.1
PROMPT ?= Explain why the sky is blue in one concise paragraph.

.PHONY: setup generate serve bench bench-122b clean-cache

setup:
	./scripts/setup-turboquant.sh

generate:
	$(PYTHON) -m turboquant_mlx.stream.stream_generate \
		--model "$(MODEL)" \
		--prompt "$(PROMPT)" \
		--max-tokens $(MAX_TOKENS) \
		--cache-budget-gb $(CACHE_GB)

serve:
	$(PYTHON) scripts/tq_openai_server.py \
		--model "$(MODEL)" \
		--cache-budget-gb $(CACHE_GB) \
		--host "$(HOST)" \
		--port $(PORT) \
		--max-tokens $(MAX_TOKENS)

bench:
	$(PYTHON) scripts/benchmark_openai.py \
		--base-url "http://$(HOST):$(PORT)/v1" \
		--model "$(MODEL)" \
		--max-tokens $(MAX_TOKENS)

bench-122b:
	$(MAKE) generate \
		MODEL="manjunathshiva/qwen3.5-122b-tq3" \
		CACHE_GB=0.25 \
		MAX_TOKENS=32 \
		PROMPT="Explain why the sky is blue in one short paragraph."

clean-cache:
	$(PYTHON) -m pip cache purge
