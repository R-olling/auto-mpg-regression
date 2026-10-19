# 溯源说明（PROVENANCE）

数据: UCI Auto MPG (Quinlan, 1993, DOI 10.24432/C5859H)
  └─ v1 (tag data-v1, md5 4f2321b47a64d9849f9325d64ceb5361) 原始 CSV，398 行
  └─ v2 (tag data-v2, md5 3e4175cf1366fa913048bf2678166b78) ml/data_pipeline.py
     清洗制品，397 行（用途：记录清洗口径；训练不照搬其全数据 IQR 删行）

代码: git commit（每次 run 的 tag code_version 可查）

评估: 从 v1 固定身份划分（6 条缺失行一律入训练折；313 条完整训练 + 6 条填补
  = 319 训练 / 79 测试）；中位数与缩放都在 Pipeline 内按折拟合；
  linear 与 Ridge 各 α 均用训练折 5 折交叉验证选型，测试集只作对照

模型: MLflow Registry AutoMPG_LinearRegression v6
  └─ run: improved-v2-linear-export
  └─ 导出: app/model/linear_regression.pkl（2105 字节）

服务: app/main.py 启动时 joblib.load 该文件；MODEL_VERSION 环境变量标识版本；

复现:
  git checkout <commit> && dvc pull
  python ml/train.py --data data/auto_mpg_v1.csv --stage improved \
      --model-type linear --scale --export app/model/linear_regression.pkl