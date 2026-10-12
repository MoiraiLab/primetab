# 数据

本仓库**不分发**原始实验数据与模型权重，仅说明来源与获取方式。

## 公开实验数据
PrimeTab 的参考样本来自公开 Prime Editing 研究的补充表：

| 来源 | 内容 | 获取 |
|---|---|---|
| DeepPrime (Yu et al., Cell 2023) | 多细胞、多编辑系统的 pegRNA 设计与编辑率 | https://doi.org/10.1016/j.cell.2023.03.034 |
| PRIDICT2 (Mathis et al., Nat Biotechnol 2024) | 多类型编辑设计与实验测量 | https://doi.org/10.1038/s41587-024-02268-2 |

使用时统一为：目标序列、预期编辑、PBS、RTT、细胞、编辑器条件；编辑率归一到 0–1 小数，
并保留原始来源标识以便追溯。特征处理（填补/标准化）参数在源参考数据上拟合。

## 模型权重
TabPFN v3 回归权重（Prior Labs）不随仓库分发。通过环境变量提供路径：
- `TFM_TABPFN_WEIGHTS=/path/to/tabpfn-v3-regressor-v3_default.ckpt`，或
- `TABPFN_MODEL_CACHE_DIR=/path/to/cache`

官方获取：https://github.com/PriorLabs/TabPFN 、https://huggingface.co/Prior-Labs/tabpfn_3

## 本地目录约定
`primetab.paths` 通过环境变量定位数据根：
- `TFM_REPOSITORY_ROOT`（仓库根，默认为包的上一级）
- `TFM_DATA_ROOT`（默认 `<repo>/data`）

> 开源前请确认 DeepPrime / PRIDICT2 补充数据的再分发许可；如需附带样例数据，仅放置
> 小规模、许可明确的子集。
