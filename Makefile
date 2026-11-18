.PHONY: docs-assets docs-check docs-serve docs-verify

docs-assets:
	python3 tools/build_portfolio_assets.py

docs-check:
	python3 tools/check_docs.py

docs-verify: docs-assets docs-check

docs-serve:
	python3 -m http.server 4173 --bind 0.0.0.0 --directory .
