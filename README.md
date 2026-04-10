# 换流变内部局部放电数据处理与特征提取项目

本项目提供一套可扩展的 Python 代码，用于完成：

1. 多格式局部放电样本读取
2. 信号预处理（去直流、滤波、去噪、归一化、平滑、异常值处理）
3. 时域/频域/时频域特征提取
4. 特征矩阵构建与保存
5. 基础可视化

## 项目结构

- `data_loader.py`：读取 CSV/Excel/TXT/NumPy，统一为 `SignalSample`
- `preprocess.py`：预处理函数与 pipeline
- `feature_extraction.py`：时域、频域、小波、小波包、STFT 特征
- `feature_matrix.py`：构建/标准化/保存特征矩阵
- `visualization.py`：波形、频谱、能量图、相关性热图、分布图
- `main.py`：端到端示例流程

## 环境依赖

建议 Python 3.9+

```bash
pip install numpy pandas scipy matplotlib scikit-learn pywavelets openpyxl
```

## 输入数据格式要求

默认场景：**每个文件对应一个样本**。

支持格式：
- `.csv`
- `.txt`
- `.xls/.xlsx`
- `.npy/.npz`

可支持：
- 单通道 `(n_samples,)`
- 多通道 `(n_samples, n_channels)`

注意事项：
- 若为表格文件，建议信号列为数值型；可通过 `signal_columns` 指定信号列。
- 若标签在文件中，可通过 `label_column` 指定，自动拼接到特征表。
- 采样频率 `fs` 需要由用户提供（单位 Hz）。

## 快速运行

1. 将样本文件放入 `./data` 目录。
2. 在 `main.py` 中设置 `fs`（采样频率）。
3. 运行：

```bash
python main.py
```

输出目录：`./outputs`
- `pd_features_raw.csv`
- `pd_features_standardized.csv`
- 可视化图片若干

## 参数默认推荐值（可调整）

- 采样频率：`fs=1_000_000 Hz`（示例值，实际请替换）
- 带通滤波：`10 kHz ~ 300 kHz`，4 阶 Butterworth
- 小波基：`db4`
- 小波分解层数：`4`
- 小波包层数：`3`
- STFT：`nperseg=256`, `noverlap=128`

调整方式：
- 预处理参数：修改 `main.py` 中 `FilterConfig` 与 `preprocess_pipeline` 参数。
- 频带划分：修改 `feature_extraction.py` 中 `frequency_domain_features` 的 `band_edges`。
- 小波相关参数：`extract_all_features(..., wavelet, wavelet_level, wpt_level)`。

## 后续扩展建议

1. PCA 降维
   - 对 `pd_features_standardized.csv` 的数值特征使用 `sklearn.decomposition.PCA`。

2. 特征筛选与重要性分析
   - Filter：方差阈值、互信息、相关性去冗余
   - Embedded：RandomForest/XGBoost 特征重要性
   - Wrapper：RFE

3. 状态分类
   - 若有标签：SVM、RandomForest、XGBoost、LightGBM
   - 评价指标：F1、AUC、PR-AUC、Kappa

4. 耐受能力评估
   - 分类路径：健康/预警/风险分级
   - 回归路径：耐受指标预测
   - 综合评价：可组合 AHP/熵权-TOPSIS/模糊评价，与数据驱动模型融合

5. 时频增强
   - 可接入 EMD/HHT（PyEMD）提取 IMF 能量、瞬时频率统计特征

## 备注

本项目特征设计强调：
- 对脉冲异常敏感（峰值、峰值因子、脉冲因子、峭度等）
- 兼顾频谱结构（主频、能量分布、熵）
- 融合时频信息（DWT/WPT/STFT）
- 便于后续论文分析与机器学习建模
