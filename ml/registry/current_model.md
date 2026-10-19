# 生产模型登记卡（current_model）

- 注册模型名: AutoMPG_LinearRegression
- 生产版本号: version 6（对应 run: improved-v2-linear-export）
- 模型形式: sklearn Pipeline(SimpleImputer -> StandardScaler -> LinearRegression)
- 评估方法: 固定身份划分（79 条独立测试记录）；中位数/缩放均在 Pipeline 内
  按训练折拟合（无泄漏）；选型依据训练折 5 折交叉验证
- 训练数据: data/auto_mpg_v1.csv（DVC v1, md5 4f2321b47a64d9849f9325d64ceb5361）；
  训练行数 319 = 313 条完整记录 + 6 条马力缺失（训练折中位数 95.0 填补）
- 代码版本: 见 run 的 tag code_version（git commit SHA）
- 导出文件: app/model/linear_regression.pkl（随 app/ 进入 Docker 镜像，约 2.1KB）
- 指标（79 条独立测试集）: RMSE 3.2691 / MAE 2.4890 / R² 0.7906
  （诚实对照：baseline 为 3.2727 / 2.4198 / 0.7902，RMSE 与 R² 持平略好、MAE 略差）
- 加载方式: joblib.load(app/model/linear_regression.pkl)，输入须为带列名的 DataFrame
  （列序见 app/features.py 的 FEATURE_COLUMNS）