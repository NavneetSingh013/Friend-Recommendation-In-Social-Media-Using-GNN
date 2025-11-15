# Makefile for Friend Recommendation GNN Project

.PHONY: help install data train evaluate demo clean test

help:
	@echo "Friend Recommendation GNN Project"
	@echo ""
	@echo "Available targets:"
	@echo "  make install     - Install dependencies"
	@echo "  make data        - Download and preprocess datasets"
	@echo "  make train       - Train all models"
	@echo "  make evaluate    - Evaluate all models"
	@echo "  make demo        - Run Streamlit demo app"
	@echo "  make test        - Run unit tests"
	@echo "  make clean       - Clean generated files"

install:
	pip install -r requirements.txt

data:
	python scripts/download_and_prepare.py --dataset synthetic --preprocess
	@echo "Data prepared. To download Facebook dataset:"
	@echo "  python scripts/download_and_prepare.py --dataset facebook --download --preprocess"

train:
	python scripts/train.py --model graphsage --dataset synthetic --config configs/graphsage_config.yaml
	python scripts/train.py --model gat --dataset synthetic --config configs/gat_config.yaml

evaluate:
	python scripts/evaluate.py --model graphsage --checkpoint data/checkpoints/graphsage/best_model.pt --dataset synthetic --config configs/graphsage_config.yaml
	python scripts/evaluate.py --model gat --checkpoint data/checkpoints/gat/best_model.pt --dataset synthetic --config configs/gat_config.yaml

demo:
	streamlit run demo/streamlit_app.py

test:
	pytest tests/ -v

clean:
	rm -rf data/processed/*
	rm -rf data/checkpoints/*
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

