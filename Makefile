PYTHON := python3
TESTS = src/quadtree.py src/histopyramid.py
MAIN = src/main.py

.PHONY: test

test: $(TESTS)
	$(foreach path,$^,$(PYTHON) $(path);)
