# TL-ANDI 流水线

这里实现论文 *Context-Constrained Transfer Learning for Tabular Foundation Models via Data Distillation* 的 TL-ANDI 方法。作者没有提供公开代码仓库，因此本实现依据论文算法与补充材料编写。

流水线执行以下步骤：

1. 将少量目标域样本按 1:1 分为校准集和验证集；
2. 用目标域模型估计每个源样本与目标任务的预测差异；
3. 用 Epanechnikov 核平滑源标签；
4. 根据查询点距离和预测差异，贪心选择不超过上下文预算的源样本；
5. 在验证集上选择核带宽和差异惩罚系数，并把“只用目标数据”作为安全候选；
6. 固定选中的源上下文，用全部目标样本重新校准残差后预测。

流水线只依赖 `model_factory(task)`，因此实验可以选择 TabPFN3、TabFM、LimiX-16M，或轻量的 `ridge` 调试后端。模型源码和权重路径仍由 `configs/tfm_backends.json` 管理。

二分类按照论文的连续条件均值算法处理：0/1 标签、核平滑后的概率标签和残差都交给回归后端，最终结果限制在 0 到 1 之间。这样不会把连续的蒸馏标签误交给只接受离散类别的分类器。

```python
from primetab.backends import create_model_factory
from primetab.retrieval.tl_andi import TLANDIConfig, TLANDIPipeline

factory = create_model_factory({"name": "tabpfn3", "kwargs": {"n_estimators": 1}})
pipeline = TLANDIPipeline(
    model_factory=factory,
    config=TLANDIConfig(task="regression", context_budget=500),
)
result = pipeline.run(X_source, y_source, X_target, y_target, X_query)
```

`bandwidth_sample_size` 控制带宽分位数估计所用的源样本数。默认抽取 2,000 行以控制内存；设为 `null` 会计算完整两两距离，只适合较小数据集。
