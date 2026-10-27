# 原生模型参考源码

这些文件从原模型目录移动而来，源码内容未修改。运行时实现位于上级目录的
`backbone_qwen35.py`、`generator_wan21.py` 等文件；生产代码不得导入本目录。
数值对照测试可以导入参考类。

移动前后 SHA-256：

| 文件 | SHA-256 |
|---|---|
| modeling_qwen3.py | `337f69441cd64a7ec2003f0fabba588fdce5bbac622351ff3ce942da1c0c4de4` |
| modeling_qwen35.py | `5f5f0b73b20afe340f65faf7edfcf05c3aacc57ac925542e82b4ac43b530d400` |
| modeling_wan21.py | `2a2787dea65d464ceefec90dcc7df6aed26e5fc839ab45393df03a28bad32110` |
