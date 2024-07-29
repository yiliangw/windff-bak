.PHONY: help
help:
	@echo "Usage: make [target]"

.PHONY: init_env
init_env:
	conda activate stgraph

.PHONY: train
train:
	python main.py --gpu_id 0 --random false --only_useful true --train_days 3 --val_days 2 --epoch 1 --var_len 5 --graph_type geo --add_apt false --loss FilterHuberLoss --data_diff false