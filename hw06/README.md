# HW06：受迫阻尼非线性物理摆

完整计算报告见 **[report.md](report.md)**，原始题目见本目录 PDF。

## 运行

```bash
python -m pip install -r requirements.txt
python pendulum.py
python -m unittest -v test_pendulum.py
```

需 Python 3.10 或更新版本，首次运行 Numba 会进行编译。运行脚本会重新生成 `results/` 下的全部数据和图像，不依赖 PDF 解析器。

## 文件

- `pendulum.py`：Euler–Cromer 积分、精确相位插值采样、五组实验、两级放大和盒计数。
- `test_pendulum.py`：更新顺序、角度约化、采样与跨边界插值、暂态舍弃、盒计数测试。
- `report.md`：中文计算报告（含全部图像）。
- `results/01_initial_conditions.*`：三组初值比较。
- `results/02_time_steps.*`：三种步长比较（各取 10000 点）。
- `results/03_magnification.*`：全局图及两级局部放大（50000 点）。
- `results/04_local_time_steps.*`：局部步长比较。
- `results/05_box_counting.*`：盒计数双对数图。
- `results/ic0.csv`、`ic1.csv`、`ic2.csv`、`dt02.csv`、`dt01.csv`：庞加莱点，列为 `theta,omega`。
- `results/box_counts.csv`、`summary.json`：盒计数及实验参数和数值结果。
- `assignment.txt`：原 PDF 的文字提取，供核对要求。

所有图同时保存 PNG 和 PDF；报告中图片使用相对路径，阅读时请保留 `results/` 目录。
