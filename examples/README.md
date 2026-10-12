# Examples

## 1. 仅生成候选（无需模型权重）
```bash
python -m primetab.generation batch \
  --input examples/pe2_generator_tasks_example.csv \
  --output-dir out/generation --nicking --ngs-primers
```
`pe2_generator_tasks_example.csv` 每行一个任务，唯一必填列 `editseq`（PRIDICT2 标记法，
如 `(G/C)`、`(+AC)`、`(-TT)`，编辑两侧各 ≥99 nt）。可选列：`pbs_lengths`（默认 `7-15`）、
`rt_overhang_lengths`（默认 `3-19`）、`max_edit_distance`、`max_edit_length`、
`include_nicking`、`include_ngs_primers`。

## 2. 生成 + 排序（需 TabPFN 权重与参考上下文）
```bash
python -m primetab.ranking \
  --input examples/pe2_generator_tasks_example.csv \
  --context /path/to/preselected_context.parquet \
  --preprocessing-reference /path/to/source_reference.parquet \
  --backend tabpfn3 --device cpu --context-budget 1000 \
  --output-dir out/ranking
```
上下文表需含 `F1_sequence_v1` 的全部 73 列及 `editing_efficiency_fraction` 标签。
TabPFN 权重通过 `--model-path` 或环境变量 `TFM_TABPFN_WEIGHTS` / `TABPFN_MODEL_CACHE_DIR`
提供（不随仓库分发，见 `../docs/method.md`）。

## 3. Python API
```python
from primetab.generation import EditRequest
from primetab.ranking import GeneratorRankerPipeline, SharedContextTFMRanker

ranker = SharedContextTFMRanker("tabpfn3", backend_kwargs={"device": "cpu"})
ranker.fit(context_features, context_labels, preprocessing_reference=source_features)
pipeline = GeneratorRankerPipeline(ranker)
result = pipeline.run([EditRequest("example", editseq)])
ranked = result.ranked_candidates
```
