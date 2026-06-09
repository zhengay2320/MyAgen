PYTHON ?= python

.PHONY: install test unittest lint-basic smoke synthetic api mcp clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

unittest:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"

lint-basic:
	$(PYTHON) -m compileall rs_service rs_mcp

smoke:
	$(PYTHON) scripts/smoke_test.py

synthetic:
	$(PYTHON) scripts/create_synthetic_geotiff.py --output workspace/synthetic.tif

api:
	$(PYTHON) -m rs_service.api

mcp:
	$(PYTHON) -m rs_mcp.server

clean:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('workspace').glob('job_*')]"
