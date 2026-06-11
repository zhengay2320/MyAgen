PYTHON ?= python

.PHONY: install test unittest lint-basic dev-check smoke synthetic api mcp clean clean-workspace

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

unittest:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"

lint-basic:
	$(PYTHON) -m compileall rs_service rs_mcp

dev-check:
	$(PYTHON) scripts/dev_check.py

smoke:
	$(PYTHON) scripts/smoke_test.py

synthetic:
	$(PYTHON) scripts/create_synthetic_geotiff.py --output workspace/synthetic.tif

api:
	$(PYTHON) scripts/run_api.py --host 127.0.0.1 --port 8765

mcp:
	$(PYTHON) scripts/run_mcp.py

clean:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('workspace').glob('job_*')]"

clean-workspace:
	$(PYTHON) -c "import shutil, pathlib; root=pathlib.Path('workspace'); [shutil.rmtree(root/name, ignore_errors=True) for name in ['outputs','previews','reports','jobs','cache','smoke']]; [p.unlink(missing_ok=True) for p in root.glob('synthetic*.tif')]; root.mkdir(exist_ok=True)"
