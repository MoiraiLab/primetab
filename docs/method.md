# PrimeTab 方法说明

本文件概述 PrimeTab 的模型、特征、检索与评测方法。细节以源码为准。

## 1. 总览
PrimeTab 把 pegRNA 设计表示为表格特征，逐候选检索参考样本，再由 TabPFN（表格基础模型）
以上下文学习预测编辑效率并排序。模型权重固定，新增实验数据只进入上下文，实现免训练适配。

## 2. 候选生成（`primetab.generation`）
无模型 PE2-NGG 生成器：依据目标序列与预期编辑枚举 PE2 设计，NGG PAM，PBS 7–15 nt、
RTT 同源延伸 3–19 nt。输出引导序列、PBS、RTT、骨架、全长 pegRNA、可选 PE3/PE3b guide、
NGS 引物与结构化错误。候选枚举逻辑改编自 PRIDICT2（见 `primetab/generation/NOTICE`、
`PROVENANCE.json`）。

## 3. 结构化特征（`primetab.features`）
把不同来源数据统一到同一特征空间。特征类别：
- 编辑位置：编辑类型、长度、距切口位置、同源延伸。
- 序列组成：PBS/RTT 长度、GC 比例、碱基比例、连续碱基。
- 局部背景：编辑前后 PAM、切口与编辑两侧碱基。
- 热稳定性：PBS/RTT 熔解温度（Biopython 近邻模型）。
- 二级结构：spacer/PBS/RTT/延伸区最低自由能（ViennaRNA，37 ℃）。

73 个原始字段，其中 6 个短 DNA 字段展开为碱基通道，最终 107 维数值输入。缺失值填补与
标准化参数在源参考数据（HEK293T）上拟合，应用于目标细胞与新候选。

## 4. 逐候选检索 + TabPFN（`primetab.retrieval` / `primetab.backends` / `primetab.inference`）
TabPFN 读取一组带实验测量的参考样本与待预测候选特征，通过上下文学习输出预测。权重固定，
每条候选使用独立的参考集（目标细胞实测近邻 + 源池分层/近邻/联合近邻）。检索实现见
`primetab.retrieval`（TL-ANDI 蒸馏与贪心锚点）。

TabPFN 推理使用标准 `tabpfn` 包（`primetab.inference.TabPFNEngine`）：对每条候选固定一组
参考上下文，拟合后预测，不跨上下文复用拟合变换。

TabPFN v3 权重不随仓库分发，通过 `TFM_TABPFN_WEIGHTS` 或 `TABPFN_MODEL_CACHE_DIR` 提供，
获取方式见 Prior Labs 官方仓库 https://github.com/PriorLabs/TabPFN 。

## 5. 一键流水线（`primetab.ranking`）
`editseq (CSV) → 生成 → 73 特征 → 排序器打分 → 每请求内按分数降序`。输出候选表、特征表、
排序结果、可选 guide/引物、错误表与带校验值的 `manifest.json`。

## 6. 跨细胞 / 跨域评测（`primetab.benchmarks`）
以源池（如 HEK293T）为参考，在多种目标细胞上按位点组（group）划分做留一/适配评测，
多随机种子，指标为 Spearman 与 MAE。注：开源版的评测 runner 只保留与 TabPFN 逐候选检索
及通用基线（XGBoost/MLP/DeepPrime 分组 CV 等）相关的路径；原项目中的域自适应机制
（OT 对齐、软域提示、残差门控等）不在开源范围。

## 7. 可解释性（`primetab.diagnostics`）
固定其他变量、扰动单一特征，计算预测响应（雅可比/响应幅度）并分析特征交互，把预测与
具体生物学因素关联。

## 8. 数据
公开实验数据（DeepPrime、PRIDICT2 补充表）不随仓库分发，获取见 `data/README.md`。
