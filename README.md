# Model Sharding

![Hugging Face](https://cdn-images-1.medium.com/max/1250/0*V0xz1iMvZqQ2mx_l.png)

## Introduction

Large Language Models (LLMs) represent a significant advancement in artificial intelligence and natural language processing.

Models such as OpenAI's GPT (Generative Pre-trained Transformer) series, Google's Gemini, PaLM, T5, and many such open-source models have achieved remarkable capabilities in understanding and generating human-like text.
However, as these models grow larger to improve performance, they also pose challenges in terms of scalability, resource requirements, and ethical considerations.

A major challenge is using such models. Leave alone using the LLM in Colab, Kaggle notebook, or locally with less amount of RAM, even loading such huge models needs high RAM which is not a feasible solution.

So one such solution will be model sharding which converts the huge models into smaller chunks which in turn takes less time and consumes less hardware for loading such huge models.

Here we will discuss model sharding using Open Source LLM Mistral 7B freely hosted on HuggingFace Platform.

## Lesser RAM

```python
%%time
model_name = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16
)
```

```
CPU times: user 36.8 s, sys: 48.5 s, total: 1min 25s
Wall time: 3min 30s
```

![Before Sharding](https://cdn-images-1.medium.com/max/1250/1*Mf3ZQ7ShygBWk5jq15mnWw.png)

```python
%%time
model_name = "Sharathhebbar24/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16
)
```

```
CPU times: user 23 s, sys: 48.7 s, total: 1min 11s
Wall time: 1min 49s
```

![After Sharding](https://cdn-images-1.medium.com/max/1250/1*PxgEovo177n3vEJ-2enpzw.png)

## References
- HF Docs: https://huggingface.co/docs/transformers/en/big_models
- Using Accelerate: https://huggingface.co/docs/transformers/en/main_classes/model#large-model-loading
- Template: https://colab.research.google.com/drive/18z7PzYkRuYfZd1Vqtz1ARkZOiwAmd9Hw?usp=sharing
- Github: https://github.com/SharathHebbar/Model-Sharding
- Medium: https://medium.com/@sharathhebbar24/llm-model-sharding-55102ecb1823
- Credits: https://medium.com/@jain.sm/sharding-large-models-for-parallel-inference-ee19844cc44#:~:text=Memory%20Efficiency%3A%20Sharding%20enables%20running,parts%2C%20reducing%20memory%20requirements%20significantly.