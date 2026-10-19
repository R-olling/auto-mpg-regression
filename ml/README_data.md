# Auto MPG 数据集说明（README_data）

## 1. 数据来源
- UCI Machine Learning Repository: Auto MPG
- 页面: https://archive.ics.uci.edu/dataset/9/auto+mpg (DOI: 10.24432/C5859H)
- 引用: Quinlan, R. (1993). Auto MPG [Dataset]. UCI ML Repository.
- 文件: auto-mpg.data（无表头、空白分隔，398 行 × 9 列）

## 2. v1 → v2 的变化（全部由 ml/data_pipeline.py 完成）

| 变化 | 说明 |
|---|---|
| 格式 | 无表头原始文件 → 带表头的规范 CSV（v1） |
| 缺失值 | v1 中 horsepower 的 6 个 "?" → v2 用中位数 93.5 填补（保留 6 行，未删行） |
| 类别编码 | origin(1/2/3) → origin_1/origin_2/origin_3 one-hot 三列 |
| 异常值 | 按 IQR 准则（区间 [0.25, 46.25]）剔除 mpg 异常行 1 行 |
| 行数 | 398 → 397 |

> 说明：v2 是"数据制品"，中位数与 IQR 阈值均在全部数据上计算，用于记录清洗口径。
> 训练时不照搬 v2：训练在 v1 上做固定划分，中位数只在训练折上拟合（见 ml/train.py）。

## 3. commit hash 对应关系

| 版本 | git tag | git commit SHA | 数据文件 | .dvc md5 |
|---|---|---|---|---|
| v1 | data-v1 | 479ccd3d8dd8488c22f41c83233d15af82cbce6c | data/auto_mpg_v1.csv | 4f2321b47a64d9849f9325d64ceb5361 |
| v2 | data-v2 | 8cdb2bff23b7c1d87a3dd0357a0bb4b85f9c434e | data/auto_mpg_v2.csv | 3e4175cf1366fa913048bf2678166b78 |

> 复现方式：`git checkout data-v1`（或 `data-v2`）后执行 `dvc pull`，即可取回对应版本数据。
> 注意：git commit SHA 与 .dvc md5 是两份独立指纹——SHA 定位"代码与指针"，md5 定位"数据内容"，
> 二者一起才能回答"哪个模型吃了哪版数据"。