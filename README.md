# Install MLPerf
use python_venv(python 3.10.12) at `$TT_METAL_HOME` which created by `./create_venv.sh`

## Git Clone

```
cd $TT_METAL_HOME
source env_set.sh
cd ..
git clone git@github.com:bos-semi/mlperf.git
```

## Install dependency

```
pip install pybind11
pip install mlc-scripts
```

## Install loadgen

```
# install loadgen (at $TT_METAL_HOME/..)
cd mlperf/loadgen
python setup.py develop --user
```

## Step for Vision Model

### Install
Install vision(classification, detection) model

```
# (at $TT_METAL_HOME/..)
cd mlperf/vision/classification_and_detection
python setup.py develop
```

### Dataset Preparation

#### ImageNet

```
# (at $TT_METAL_HOME/../mlperf)
mlcr get,dataset,imagenet,validation --outdirname=./data -j
```

```
python3 -c "
import os

data_dir = './data/imagenet-2012-val/'
output_path = os.path.join(data_dir, 'val_map.txt')

# ILSVRC2012 val 레이블 (1-indexed synset → 0-indexed class)
# 공식 ground truth: https://github.com/tensorflow/models/blob/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt
import urllib.request
label_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt'
synset_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_lsvrc_2015_synsets.txt'

synsets_raw, _ = urllib.request.urlretrieve(synset_url)
labels_raw, _ = urllib.request.urlretrieve(label_url)

with open(synsets_raw) as f:
    synsets = [l.strip() for l in f.readlines()]
synset_to_idx = {s: i for i, s in enumerate(synsets)}

with open(labels_raw) as f:
    val_labels = [l.strip() for l in f.readlines()]

files = sorted([f for f in os.listdir(data_dir) if f.endswith('.JPEG')])
with open(output_path, 'w') as out:
    for fname, synset in zip(files, val_labels):
        idx = synset_to_idx.get(synset, 0)
        out.write(f'{fname} {idx}\n')

print(f'Generated {len(files)} entries -> {output_path}')
with open(output_path) as f:
    print('Sample:', f.readlines()[:3])
"
```

#### COCO2017

```
# (at $TT_METAL_HOME/../mlperf)
mlcr run --tags=get,dataset,object-detection,coco,_val,_2017 --to=./data/coco
```

### Run Benchmark

run below command at `/vision/classification_and_detection` (if you add `--accuracy`, then it provides acc/mAP/…)

#### ResNet50

```
python python/main.py --profile resnet50-ttnn-trace-2cq --scenario Offline --dataset-path mlperf/data/imagenet-2012-val/ --model dummy --threads 1
```

#### ViT

```
python python/main.py --profile vit-ttnn-trace-2cq --scenario Offline --dataset-path mlperf/data/imagenet-2012-val/ --model dummy --threads 1
```

#### YOLOv8s

```
python python/main.py --profile yolo-ttnn-trace --scenario Offline --dataset-path mlperf/data/coco/ --model dummy --threads 1
```

## Step for Language Model

### Install
Install language(LLM, VLM) model

```
pip install git+https://github.com/CentML/pydantic-typer.git@wangshangsam/preserve-full-annotated-type
pip install pympler typer hiclass rapidfuzz
pip install typing_extensions
pip install "openai[aiohttp]"
```

### Dataset Preparation

#### llama3-1-8b-sample-cnn-eval-5000.uri

```
cd mlperf/data
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://inference.mlcommons-storage.org/metadata/llama3-1-8b-sample-cnn-eval-5000.uri
```

#### shopify-catalogue

```
# automatically downloaded at run benchmark
```

### Run Benchmark

#### Llama3.1-8B

run below command at `/language/llama3.1-8b`

```
python -u main.py --scenario SingleStream --model-path meta-llama/Llama-3.1-8B-Instruct --dataset-path mlperf/data/sample_cnn_eval_5000.json --user-conf mlperf/language/llama3.1-8b/user.conf --vllm
```

### Qwen2.5-VL-7B

run below command at `/multimodal/qwen3-vl/src`

```
HF_MODEL=Qwen/Qwen2.5-VL-7B-Instruct python -m mlperf_inf_mm_q3vl.cli benchmark vllm --settings.test.scenario 1 --settings.test.mode 1 --vllm.model.repo_id Qwen/Qwen2.5-VL-7B-Instruct --vllm.model.revision main --vllm.cli=--tensor-parallel-size=1 --vllm.cli=--no-enable-prefix-caching --vllm.cli=--max-model-len=8192 --vllm.cli=--mm-processor-kwargs='{"max_pixels":827904}' --vllm.cli=--max-num-seqs=1 --vllm.cli=--block-size=32 --vllm.cli=--override_tt_config='{"enable_model_warmup": false}' --settings.user_conf.path mlperf/multimodal/qwen3-vl/src/user.conf
```
