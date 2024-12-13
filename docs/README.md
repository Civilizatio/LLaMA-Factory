# 实验环境配置和相关文档说明

## Llama-Factory 安装

这里主要参考官方文档的说明，具体可以参考[官方文档](../README.md)。首先创建 conda 环境，满足文档中版本需求，下载这个后直接执行

```bash
pip install -e ".[torch,metrics]"
```

## 数据集准备

数据集主要放在 `data` 目录下，这里主要是为了方便管理，可以根据自己的需求进行修改。数据集的准备可以参考[数据集准备](../data/README.md)。

这里我们需要使用的是 `alpaca_data_cleaned` 数据集，我已经下好了。里面有 5w 多条对话，由于我们的要求是 1w 条，其中 7k 条训练数据，2k 条验证数据，1k 条测试数据。所以我们需要对数据进行处理。这里参见 `scripts` 目录下的 `my_dataprocess.py` 文件，可以直接运行，将在 `data` 目录下生成 `alpaca_data_train.json`、`alpaca_data_valid.json`、`alpaca_data_test.json` 三个文件，后面可以直接使用。

> 这里还需要在 `data` 目录下的 `dataset_info.json` 文件中添加相应信息，我这里也已经添加好了。因为是符合官方的数据集格式，所以只需要添加 `alpaca_data_train.json`、`alpaca_data_valid.json`、`alpaca_data_test.json` 三个文件的路径即可。

``` json
"alpaca_cleaned":{
    "file_name": "alpaca_data_cleaned.json"
  },
  "alpaca_cleaned_train":{
    "file_name": "alpaca_data_train.json"
  },
  "alpaca_cleaned_valid":{
    "file_name": "alpaca_data_val.json"
  },
  "alpaca_cleaned_test":{
    "file_name": "alpaca_data_test.json"
  },
```

## 模型准备

我们这里使用的是 `Llama-3.2-1B-Instruct` 模型，与原来的 `Llama-3.2-1B` 模型相比，增加了 `Instruct` 的功能，可以根据我们的指令进行对话。

### 模型下载
    
1. 通过 `huggingface` 官方下载，这里需要注册账号，然后由于该模型为 `Gated Model`，需要有 `Access Token`，
申请时填写美国一个地址就行。然后生成一个 `Access Token`，然后使用该 `Access Token` 下载模型。用下面的命令：

```bash
huggingface-cli download --resume-download meta-llama/Llama3.2-1B-Instruct --local-dir {本地地址} --token hf_***
```

在服务器上这种方法比较推荐，因为下载比较快。

2. 也可以从我这里传过去。

### 测试模型

首先，我们需要进入到 `Llama-Factory` 的根目录下，然后进入 [`template.py`](../data/template.py) 文件，修改 `llama3` 模板，主要是参考模型目录下的 `special_tokens_map.json` 文件，将里面的 `special_tokens` 替换到 `template.py` 文件中，否则生成会有问题。

```python
_register_template(
    name="llama3",
    format_user=StringFormatter(
        slots=[
            (
                "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_system=StringFormatter(slots=["<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"]),
    format_observation=StringFormatter(
        slots=[
            (
                "<|start_header_id|>tool<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<|eot_id|>"],
    replace_eos=True,
    replace_jinja_template=False,
)
```

然后修改对应的配置文件，例如 `examples/inference/llama3.yaml`：（只需要将模型目录换为自己的就行）
  
```yaml
model_name_or_path: /home/like/Models/Llama-3.2-1B-Instruct
template: llama3
```

然后调用 `llamafactory-cli` 进行推理，主要有三种形式：

1. `chat`：对话形式，在命令行中进行对话。
```bash
llamafactory-cli chat --config examples/inference/llama3.yaml
```
2. `webchat`：网页对话形式，可以在浏览器中进行对话。
```bash
llamafactory-cli webchat --config examples/inference/llama3.yaml
```
3. `api`：接口形式，可以通过指定接口进行对话。
```bash
API_PORT=8000 CUDA_VISIBLE_DEVICES=0 llamafactory-cli api --config examples/inference/llama3_vllm.yaml
```
最后一种的配置文件如下：
```yaml
model_name_or_path: /home/like/Models/Llama-3.2-1B-Instruct
template: llama3
# infer_backend: vllm 
# vllm_enforce_eager: true
adapter_name_or_path: saves/llama3-1b/lora/sft
finetuning_type: lora
```
我这里是使用了 sft 后的模型推理，因此需要指定 `adapter_name_or_path` 和 `finetuning_type`。原始模型不需要。然后就是 `infer_backend`，这里是使用 `vllm`，这个是 `VLLM` 的推理方式，可以加速推理，但是我的服务器上下不好 `vllm`，所以注释掉了。默认为 `huggingface` 的推理方式。

在相应端口开启服务后，如我上面指定了 `8000` 端口，可以通过 `http://0.0.0.0:8000/v1` 进行对话。

调用接口的方式可以参考 `scripts` 目录下的 `api_call_example.py` 文件，**注意**：不可以直接运行，需要修改 `base_url` 和 `api_key` 的内容。
`base_url`：如果服务器没有用跳板机之类的，理论上就是`http://0.0.0.0:8000/v1`，其他情况另外解决。
`api_key`：这个是自动调用环境变量的 `API_KEY`，可以在服务器上设置，也可以直接运行时设置。（好像没有设置直接"0"就可以）
```bash
API_KEY="0" API_PORT=8000 CUDA_VISIBLE_DEVICES=0 llamafactory-cli api --config examples/inference/llama3_vllm.yaml
```

这个调用应该没问题，如果要换为其他 `message`，可以修改 `message` 的内容。但要注意，`message` 的格式要符合 `role`, `content` 的格式，其中 `role` 为 `user` 或 `assistant`，`content` 为对应的内容。我们这里一般就是只用 `user` 角色。

### 模型训练

这里主要参考 `examples/inference/llama3_lora_sft.yaml` 内的内容，使用命令：

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
```

里面的参数配置并不全，关于 lora 的参数也没有设置，也没有做验证。可以找更全的参数配置。

训练完后，会在 `saves` 目录下生成相应的模型，可以用于推理，此时修改 `adapter_name_or_path` 为对应的路径即可。

### 模型评估

这里我没怎么跑，调用下面命令：

```bash
llamafactory-cli eval examples/inference/llama3_pre.yaml
```

就可以跑出模型的 BLEU 和 ROUGE 分数。

我的代码中 `scripts` 目录下的 `my_inference.py` 文件，可以直接运行，但是只是将预测的结果输出到文件中，没有进行评估。


## 后续任务

1. 参考大作业要求，跑出一些评测指标
2. 改变 lora 配置，跑出更好的模型

