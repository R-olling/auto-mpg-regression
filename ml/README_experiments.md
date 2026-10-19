# 实验说明（README_experiments）

## 哪次运行适合上线生产

improved-v2-linear-export（Registry version 6）。理由：

1. **无泄漏评估**：在与 baseline 完全相同的 79 条独立测试记录上评估；中位数与缩放
   都在 Pipeline 内按训练折拟合，测试集全程不参与；
2. **指标持平略优**：RMSE 3.2691 / R² 0.7906，优于 baseline（3.2727 / 0.7902）；
   但需如实说明 MAE 略差（2.4890 vs 2.4198）——增益微乎其微且不一致；
3. **数据质量更好**：6 处 horsepower 缺失通过训练折中位数（95.0）被用上，
   origin 正确 one-hot 编码（避免把 1/2/3 当成有大小关系的数值）；
4. **形式最简单**：线性回归 + 标准化；CV 显示三个 Ridge α 与 linear 相差不足
   0.001（噪声级），按简约原则不引入额外超参；
5. **可复现**：run 记录了 code_version（git SHA）与 data_version（DVC md5）双指纹。

## 优化的指标及为什么重要

主指标 **RMSE**（均方根误差，单位 mpg）。

理由：本应用的输出是连续油耗值，RMSE 与大误差同量纲、对异常预测敏感，直接反映
"预测偏离真实油耗多少英里/加仑"。对油耗咨询类场景，大误差比典型误差更伤用户体验
（报错 5 mpg 比报错 2 mpg 糟得多），因此选 RMSE 而非 MAE 作主指标；R² 作为辅助参考
（模型解释了油耗差异的多大比例）。

诚实对照：在同一测试集上，清洗只让 RMSE 改善约 0.1%、MAE 略差，不能宣称"全面优于"；
交叉验证显示 α=10 最差（3.4013），该规模数据上强正则化无收益。

## 团队补充实验：交互特征（weight × horsepower）

角色 A（490047748-design）比较现有 improved 模型的 9 维特征与增加
`weight_x_horsepower = weight * horsepower` 后的 10 维特征。
这里的“基准特征”指现有 improved 模型，不是历史上的原始 7 维 baseline。

### 数据与公平比较

- 输入为原始 `data/auto_mpg_v1.csv`，DVC MD5 为
  `4f2321b47a64d9849f9325d64ceb5361`。脚本核验实际文件字节与 DVC 指针，
  不一致即停止。本次使用本机已有且 MD5 完全匹配的副本，未执行 Azure 下载。
- 复用 `ml.train.make_fixed_split(raw, 0.2, 42)` 与 `prepare_improved`。
  392 条马力完整记录划分为 313 条训练记录和 79 条测试记录；
  6 条马力缺失记录全部加入训练集，最终为 319 / 79。
- 两组模型共用完全相同的训练记录与测试记录，记录 ID 同时保存到 MLflow。
  两组均采用 `SimpleImputer(median) → StandardScaler → LinearRegression`。
  中位数、均值、标准差只从训练集学习，训练马力中位数为 95.0。
- 交互项在填补缺失值之前计算。因此缺马力的 6 条记录也缺交互项，
  Pipeline 对这两列分别用各自的训练中位数填补；这不等价于先填补马力再相乘。
  该顺序与分工指南的实验实现一致。

### 实测结果

| 配置 | 特征数 | 测试 RMSE（mpg） | MAE（mpg） | R² |
|---|---:|---:|---:|---:|
| 基准特征 | 9 | 3.2691 | 2.4890 | 0.7906 |
| 增加 weight × horsepower | 10 | 3.0245 | 2.3364 | 0.8208 |

在相同的 79 条测试记录上，RMSE 降低约 0.2446 mpg（相对下降 7.48%），
MAE 降低约 0.1526 mpg，R² 提高约 0.0302。
结果支持“增加该乘积特征改善了本次固定划分的预测表现”，不能单凭一次测试集对比
断言物理因果关系、统计显著性或其他数据集上的普遍收益。测试集沿用团队已有划分，
并非一份从未查看过的新外部验证集；本实验也未新增交叉验证或进行超参数搜索。

生产模型仍为现有 9 维流水线，`app/model/linear_regression.pkl` 不变。
实验模型只保存在本地 MLflow，不注册到生产模型 Registry，不触发发布。
如以后采用交互项，需要同步服务端特征构造、测试、模型注册和发布验证。

### Git Bash 复现与查看

在仓库根目录、依赖安装完成且 v1 数据准备就绪后执行：

```bash
./.venv/Scripts/python.exe ml/experiments/exp_interaction.py
./.venv/Scripts/python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

浏览器打开 `http://127.0.0.1:5000`，选择 `AutoMPG_Team_Experiments`，
比较最新且状态为 FINISHED 的 `exp-baseline_features` 与 `exp-with_interaction`。
每次重新运行会创建新的一对记录，不覆盖旧实验。脚本将数据库固定在仓库根目录，
不会因终端的 `MLFLOW_TRACKING_URI` 环境变量而把实验写到共享服务器。

每条实验记录保存以下证据：

| 内容 | MLflow 中的位置 |
|---|---|
| Git 提交 SHA、数据 MD5、工作区是否有未提交改动 | tags：`code_version`、`data_version`、`worktree_dirty` |
| 特征、随机种子、训练与测试行数、预处理方法 | params |
| RMSE、MAE、R² | metrics |
| 已拟合的预处理与线性回归流水线 | logged model `model`，地址同时记录于 `evidence/result.json` |
| 实际训练与测试记录 ID | `evidence/split.json` |
| 逐条测试预测、真实值、残差 | `evidence/test_predictions.csv` |
| 填补中位数、依赖版本 | `evidence/preprocessing.json`、`evidence/versions.json` |
| 实验与复用代码快照、依赖声明及 SHA256 | `source/`、`evidence/source_sha256.json` |

代码快照与 `worktree_dirty` 标签用于防止把尚未提交的脚本误认为已包含在 Git SHA 中。
数据文件、`mlflow.db`、`mlruns/` 与虚拟环境均由现有规则忽略，不提交到 Git。
其他组员需要在自己的环境运行脚本，才能在本机看到自己的 MLflow 记录。
