## Dependencies

```bash
conda create -n stgraph python=3.12
conda activate stgraph
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c dglteam/label/th23_cu121 dgl
```

## Training

```bash
make train
```