# YCYGL MMN

## 论文引用

张耀曾, 马静. 基于多模态多层级注意力网络的社交平台谣言检测[J]. 运筹与管理, 2025, 34(10): 17-23.

DOI: https://doi.org/10.12005/orms.2025.0303

## 项目结构

```text
ycygl_MMN
├── README.md
├── requirements.txt
├── run_pipeline.py
├── data\
│   └── README.md
├── data_id\
│   ├── twitter_tvt_list.txt
│   └── weibo_tvt_list.txt
├── gcn_ycygl_pipeline\
│   ├── __init__.py
│   ├── config.py
│   ├── prepare_dataset.py
│   ├── textgcn_features.py
│   ├── image_features.py
│   ├── model.py
│   └── train.py
└── runs\
    └── ...
```

- `data\`：保存数据来源、引用依据和复现数据结构说明，不提交完整图片和特征文件。
- `data_id\`：保存按照引用数据集来源划分整理的样本 id 清单。
- `gcn_ycygl_pipeline\`：核心 Python 包，包含数据读取、GCN 特征生成、图像特征生成、模型和训练逻辑。
- `runs\`：运行输出目录，保存中间特征、训练日志、模型权重和指标文件。
- `run_pipeline.py`：总入口，按阶段串起 GCN 特征、图像/ELA 特征和三特征融合训练。
- `requirements.txt`：最小依赖列表。

## 数据来源

Twitter：

- Dataset：MediaEval 2016 Verifying Multimedia Use
- GitHub：https://github.com/MKLab-ITI/image-verification-corpus/tree/master/mediaeval2016
- Source repository：MKLab-ITI/image-verification-corpus
- Citation：Boididou, C., Papadopoulos, S., Zampoglou, M., Apostolidis, L., Papadopoulou, O., & Kompatsiaris, Y. (2018). Detection and visualization of misleading content on Twitter. International Journal of Multimedia Information Retrieval, 7(1), 71-86.
- 划分来源：Twitter 训练集 ID 来自 `mediaeval2016/devset/posts.txt`，测试集 ID 来自 `mediaeval2016/testset/posts_groundtruth.txt`；标签映射为 `real -> 0`、`fake -> 1`。

Weibo：

- Dataset：Weibo multimodal rumor detection dataset
- GitHub：https://github.com/wangzhuang1911/Weibo-dataset
- Source repository：wangzhuang1911/Weibo-dataset
- Citation：Jin, Z., Cao, J., Guo, H., Zhang, Y., & Luo, J. (2017). Multimodal Fusion with Recurrent Neural Networks for Rumor Detection on Microblogs. ACM Multimedia 2017, 795-816.
- 划分来源：Weibo数据集来源划分的训练集、验证集、测试集
去除空文本、无意义文本、没有图片引用的数据，无法按链接顺利下载的图片等，实际使用的数据可能和清洗过程有关，数量可能存在差异。

## 数据划分清单

`data_id\` 中保存训练、验证、测试划分清单，划分沿用 `数据来源` 中对应原始数据集的划分定义。

```text
data_id
├── twitter_tvt_list.txt
└── weibo_tvt_list.txt
```

每行格式：

```text
data_id    split    label
```

- `twitter_tvt_list.txt`：Twitter / MediaEval 2016 样本划分清单，沿用 Twitter 数据来源中的原始划分，包含 `train` 和 `test`。
- `weibo_tvt_list.txt`：Weibo 样本划分清单，沿用 Weibo 数据来源中的原始划分，包含 `train`、`valid` 和 `test`。

复现时，样本文件中的 `split` 字段应与对应 `data_id` 清单一致。

## 代码文件说明

- `run_pipeline.py`：主流程脚本。读取数据配置，生成 `split.tsv`，按 `gcn`、`image`、`train` 三个阶段执行，也支持 `all` 一次跑完。
- `gcn_ycygl_pipeline\config.py`：集中配置数据路径、默认输出目录和 wandb 默认参数。
- `gcn_ycygl_pipeline\prepare_dataset.py`：读取样本文件，解析 `data_id`、文本、图片 id、标签和划分；同时提供简单分词、划分生成和 TextGCN 输入文件写出。
- `gcn_ycygl_pipeline\textgcn_features.py`：构建 TextGCN 图，训练隐藏维度为 200 的 TextGCN，并把每个样本的 GCN200 特征保存为 `{data_id}.pt`。
- `gcn_ycygl_pipeline\image_features.py`：从原图中提取 ResNet50 layer3 的 1024 维特征；同时生成 ELA 图并提取 ELA1024 特征。
- `gcn_ycygl_pipeline\model.py`：按特征投影、注意力融合、分类头组织三特征融合模型 `ThreeFeatureRumorModel`。
- `gcn_ycygl_pipeline\train.py`：按参数解析、随机种子、Dataset/DataLoader、模型准备、`train_epoch`、`eval_epoch`、`test_epoch`、训练主循环和入口函数组织训练流程，使用 `BCEWithLogitsLoss` 训练融合检测模型。
- `gcn_ycygl_pipeline\__init__.py`：包初始化文件。

## 流程

```text
样本文本 -> TextGCN -> GCN 200
干净原图 -> ResNet50 layer3 -> 原图 1024
干净原图 -> ELA -> ResNet50 layer3 -> ELA 1024
GCN200 + 原图1024 + ELA1024 -> 三特征融合检测
训练损失函数 -> BCEWithLogitsLoss
```

## 数据准备

代码默认从 `data/raw` 读取数据，也可以通过环境变量 `YCYGL_DATA_ROOT` 指定数据根目录。数据目录结构如下：

```text
data/raw/
├── weibo/
│   ├── text/
│   └── images/
└── twitter/
    ├── text/
    └── images/
```

如果不使用默认目录，在 PowerShell 中先设置数据根目录：

```powershell
$env:YCYGL_DATA_ROOT = "你的数据根目录"
```

输出目录默认是 `runs`，也可以用 `YCYGL_WORK_DIR` 修改：

```powershell
$env:YCYGL_WORK_DIR = "你的输出目录"
```

样本文件需要能解析出：

```text
data_id
text
image_id
label
split
```

其中 `label` 支持 `0/1`、`real/fake`、`nonrumor/rumor` 这类写法；`split` 使用引用数据集提供的训练集、验证集和测试集划分，支持 `train`、`valid`、`test`。

## 复现步骤

先进入项目目录并安装依赖：

```powershell
cd ycygl_MMN
pip install -r requirements.txt
```

先跑小样本 smoke test，确认数据路径、图片读取、GCN、图像特征和训练流程都能通：

```powershell
python .\run_pipeline.py --dataset weibo --limit 32 --gcn_epochs 2 --epochs 2 --no_pmi
```

确认 smoke test 能跑通后，再跑完整微博：

```powershell
python .\run_pipeline.py --dataset weibo
```

完整推特：

```powershell
python .\run_pipeline.py --dataset twitter
```

也可以分阶段复现。顺序是先生成 GCN200，再生成原图1024和 ELA1024，最后训练融合模型：

```powershell
python .\run_pipeline.py --dataset twitter --stages gcn
python .\run_pipeline.py --dataset twitter --stages image
python .\run_pipeline.py --dataset twitter --stages train
```

默认不会联网下载 ResNet50 权重。若本机允许下载或已有 torchvision 权重缓存，可以加：

```powershell
python .\run_pipeline.py --dataset twitter --image_pretrained
```

启用 wandb：

```powershell
python .\run_pipeline.py --dataset weibo --use_wandb --wandb_project gcn_ycygl_three_feature --wandb_entity your_entity
```

常用参数：

```text
--dataset           weibo 或 twitter
--stages            gcn、image、train 或 all
--limit             只取前 N 条样本，用于快速检查
--overwrite         重新生成已有特征
--device            cpu 或 cuda
--gcn_epochs        TextGCN 训练轮数
--gcn_hidden_dim    GCN 特征维度，默认 200
--epochs            三特征融合模型训练轮数
--batch_size        batch size
--use_wandb         启用 wandb
```

输出默认在：

```text
runs\weibo
runs\twitter
```

关键输出：

```text
gcn_features\{data_id}.pt
image_features\{data_id}.pt
ela_features\{data_id}.pt
three_feature_train\best_three_feature_model.pt
three_feature_train\training_log.csv
three_feature_train\test_metrics.json
```

## 输出文件说明

- `split.tsv`：本次运行使用的样本划分。
- `data\text_dataset\{dataset}.txt`：TextGCN 使用的样本文本。
- `data\text_dataset\clean_corpus\{dataset}.txt`：TextGCN 使用的分词文本。
- `gcn_features\{data_id}.pt`：每个样本的 GCN200 特征。
- `image_features\{data_id}.pt`：每个样本的原图 ResNet1024 特征。
- `ela_features\{data_id}.pt`：每个样本的 ELA ResNet1024 特征。
- `textgcn_model.pt`：TextGCN 权重。
- `three_feature_train\best_three_feature_model.pt`：三特征融合模型最优权重。
- `three_feature_train\training_log.csv`：每轮训练和验证指标。
- `three_feature_train\test_metrics.json`：测试集指标。
