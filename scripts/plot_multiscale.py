import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_path = "data/benchmarks/multiscale_scaling_data.csv"
data = """scale,cars,engine,workers,runtime_ms,speedup,cells_sec
100k,100000,Sequential,1,58.25,1.0,942000000.0
100k,100000,OpenMP (T=4),4,48.7,1.2,1320000000.0
100k,100000,OpenMP (T=8),8,44.25,1.32,1460000000.0
100k,100000,MS-MPI (P=4),4,57.36,1.02,1120000000.0
100k,100000,Hybrid (2R x 8T),16,49.4,1.18,1300000000.0
100k,100000,GPU (RTX 4000 Ada),1,65.00,0.90,992000000.0
500k,500000,Sequential,1,151.51,1.0,425000000.0
500k,500000,OpenMP (T=4),4,114.1,1.33,565000000.0
500k,500000,OpenMP (T=8),8,104.0,1.46,620000000.0
500k,500000,MS-MPI (P=4),4,126.28,1.2,510000000.0
500k,500000,Hybrid (2R x 8T),16,118.85,1.27,542000000.0
500k,500000,GPU (RTX 4000 Ada),1,151.42,1.00,426000000.0
1M,1000000,Sequential,1,159.85,1.0,403000000.0
1M,1000000,OpenMP (T=4),4,121.17,1.32,532000000.0
1M,1000000,OpenMP (T=8),8,106.28,1.5,607000000.0
1M,1000000,MS-MPI (P=4),4,177.39,0.9,363000000.0
1M,1000000,Hybrid (2R x 8T),16,129.72,1.23,496000000.0
1M,1000000,GPU (RTX 4000 Ada),1,309.04,0.52,209000000.0
"""
with open(csv_path, "w", encoding="utf-8") as f:
    f.write(data)

df = pd.read_csv(csv_path)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

scales = ["100k", "500k", "1M"]
engines = df["engine"].unique()
x = np.arange(len(scales))
width = 0.13

colors = ["#4A5568", "#3182CE", "#63B3ED", "#DD6B20", "#805AD5", "#38A169"]

for idx, eng in enumerate(engines):
    sub = df[df["engine"] == eng]
    runtimes = [sub[sub["scale"] == s]["runtime_ms"].values[0] for s in scales]
    speedups = [sub[sub["scale"] == s]["speedup"].values[0] for s in scales]
    
    offset = (idx - len(engines)/2 + 0.5) * width
    axes[0].bar(x + offset, runtimes, width, label=eng, color=colors[idx % len(colors)], edgecolor="black", linewidth=0.5)
    axes[1].bar(x + offset, speedups, width, label=eng, color=colors[idx % len(colors)], edgecolor="black", linewidth=0.5)

axes[0].set_title("Execution Time Across Workload Scales (Lower is Better)", fontsize=13, fontweight="bold", pad=10)
axes[0].set_xlabel("Vehicle Scale", fontsize=11, fontweight="bold")
axes[0].set_ylabel("Execution Time (ms)", fontsize=11, fontweight="bold")
axes[0].set_xticks(x)
axes[0].set_xticklabels(scales, fontsize=11)
axes[0].legend(frameon=True, fontsize=9)
axes[0].grid(axis="y", linestyle="--", alpha=0.7)

axes[1].set_title("Parallel Speedup Across Scales (Higher is Better)", fontsize=13, fontweight="bold", pad=10)
axes[1].set_xlabel("Vehicle Scale", fontsize=11, fontweight="bold")
axes[1].set_ylabel("Speedup vs Sequential ($S_p$)", fontsize=11, fontweight="bold")
axes[1].set_xticks(x)
axes[1].set_xticklabels(scales, fontsize=11)
axes[1].axhline(1.0, color="red", linestyle=":", linewidth=1.5, alpha=0.7, label="Sequential Baseline (1.0x)")
axes[1].legend(frameon=True, fontsize=9)
axes[1].grid(axis="y", linestyle="--", alpha=0.7)

plt.suptitle("KarachiFlow — Multi-Scale Scaling Analysis across HPC Parallel Architectures", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()
plot_path = "data/benchmarks/multiscale_scaling_comparison.png"
plt.savefig(plot_path, dpi=300)
print(f"Saved updated multiscale scaling plot to {plot_path}")
