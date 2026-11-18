# Contributing

Issues and pull requests are welcome.

```bash
git clone https://github.com/syndromics/syndromics.git
cd syndromics
pip install -e '.[dev]'
pytest
```

Please include a test with any change to `diagnose.py`, and run `benchmarks/sweep.py`
if the change could affect detection accuracy.

If you have hardware syndrome data you would like analysed, open an issue or contact us
through syndromics.ai. We never publish customer data.
