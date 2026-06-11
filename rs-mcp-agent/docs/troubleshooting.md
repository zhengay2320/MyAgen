# Troubleshooting

## rasterio / GDAL Install Fails

Prefer conda-forge on Windows:

```bash
conda install -c conda-forge rasterio gdal
```

If you only need fake tests, the project has a fallback raster container, but production GeoTIFF IO should use Rasterio/GDAL.

## GeoPandas Install Fails

Install geospatial wheels together:

```bash
conda install -c conda-forge geopandas shapely fiona pyproj
```

Without GeoPandas, GPKG writing degrades to a JSON fallback file. GeoJSON still works.

## OpenMMLab Dependency Conflicts

Use a separate environment for MMSegmentation, MMDetection, MMRotate, and Open-CD:

```bash
conda create -n rs-openmmlab python=3.10
conda activate rs-openmmlab
pip install -U openmim
mim install mmengine mmcv
```

Then install the specific package versions required by your config/checkpoint.

## MCP stdio stdout Is Polluted

MCP stdio requires stdout to contain only protocol messages. Ordinary logs must go to stderr or a file. Use:

```bash
python scripts/run_mcp.py
```

Avoid adding `print()` calls in `rs_mcp/server.py`.

## FastAPI Service Is Not Running

MCP tools call the FastAPI backend through `RS_SERVICE_URL`. Start it first:

```bash
python scripts/run_api.py --host 127.0.0.1 --port 8765
```

Then verify:

```bash
curl http://127.0.0.1:8765/health
```

## Weight Path Does Not Exist

Large model weights are not committed. Put weights under `weights/` or change `configs/models.yaml` to an absolute path.

Example:

```yaml
checkpoint: D:/models/mmseg/model.pth
```

## Windows Path Issues

Use absolute paths in MCP client configs. Escape backslashes or use forward slashes:

```toml
cwd = "D:/program_myself/Myagent/rs-mcp-agent"
RS_WORKSPACE = "D:/program_myself/Myagent/rs-mcp-agent/workspace"
```

## pytest Is Missing

Install dev dependencies:

```bash
pip install -e ".[dev]"
```

The repository also supports:

```bash
python -m unittest discover -s tests -p "test_*.py"
```
