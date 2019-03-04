PYTHON := python3
TESTS = src/quadtree.py src/hystopyramid.py
MAIN = src/main.py

.PHONY: test

test: $(TESTS)
	$(foreach path,$^,$(PYTHON) $(path);)
