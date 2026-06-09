PYTHON ?= python

.PHONY: test smoke synthetic api mcp clean

test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"

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
