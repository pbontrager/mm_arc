# MM Llama3.2 for ARC-AGI Challenge Demo
The [ARC-AGI Challenge](https://arcprize.org/) is a competation to solve the puzzles from the Abstraction and Reasoning Corpus (ARC) first outlined [here](https://arxiv.org/abs/1911.01547). This repo is a small demo project to attempt to demonstrate how multimodal (MM) understanding can help large langugae models (LLMs) improve their performance on these challenges that are all very visual.

The first approach here is very naive and simply converts the puzzles into the images that are shown to human players [here](https://arcprize.org/play) and feed Llama 3.2 Vision the questions in both text and image format and finetune it on the answers. The goal is to see if the MM approach can improve over a pure textual approach for the same sized Llama model.

## Installation

**Step 1:** [Install PyTorch](https://pytorch.org/get-started/locally/).

```
# Nightly install for latest features
pip install --pre torch torchvision torchao --index-url https://download.pytorch.org/whl/nightly/cu121
pip install --pre torchtune --extra-index-url https://download.pytorch.org/whl/nightly/cpu --no-cache-dir
```

```
# Install requirements
pip install -e .
```

**Step 2:** Download the models and datasets

```
# text only model for validation (need hf access)
tune download ...
```



## Running recipes

All recipes are in torchtune while the configs are local. Run recipes as shown below. For more tune options see [torchtune Docs](https://pytorch.org/torchtune/stable/tune_cli.html)

> [!NOTE]
> Due to [this issue](https://github.com/pytorch/torchtune/issues/1540) you can't use `tune run` from this repo


Single Device:
```bash
tune run full_finetune_single_device --config configs/11B_full_single_device.yaml
```
