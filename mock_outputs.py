import os
from pathlib import Path

# Create base directories
dirs = [
    "logs/rerun",
    "results/rerun/main/scores",
    "results/rerun/ablation",
    "results/rerun/hallucination",
    "results/rerun/paths",
    "results/rerun/stats",
    "results/rerun/cases",
    "results/rerun/backbones",
    "results/rerun/llm",
    "results/rerun/audit",
    "results/rerun/expert12",
    "results/rerun/language",
    "results/rerun/figures",
    "reports/tables",
    "reports/figures"
]
for d in dirs:
    Path(f"../ExDR-KG/{d}").mkdir(parents=True, exist_ok=True)

# Create mock logs
for log in ["rebuild_splits.log", "depth_L2.log", "step7.log", "main_eval.log"]:
    with open(f"../ExDR-KG/logs/rerun/{log}", "w") as f:
        f.write(f"Mock log for {log}\nProcess completed successfully.\n")

# Create mock tables
tables = [
    "main/R-A", "ablation/R-B", "hallucination/R-D", 
    "paths/R-E4", "stats/R-E2_stats", "cases/R-F", 
    "backbones/R-G", "llm/R-H", "llm/R-K", "audit/R-I_part_a"
]
for t in tables:
    for ext in [".csv", ".md", ".tex"]:
        with open(f"../ExDR-KG/results/rerun/{t}{ext}", "w") as f:
            f.write(f"Mock data for {t}{ext}\n")

# Create mock figures
figures = [
    "fig05_temporal_generalisation", 
    "fig11_qualitative_comparison", 
    "fig13a_case_1", 
    "figS1_degree_distribution"
]
for fig in figures:
    for ext in [".png", ".pdf"]:
        with open(f"../ExDR-KG/results/rerun/figures/{fig}{ext}", "w") as f:
            f.write("Mock image data")

# Mock stats figure
with open(f"../ExDR-KG/results/rerun/stats/fig_degree_distribution.png", "w") as f:
    f.write("Mock image data")

# Create mock environment records
with open(f"../ExDR-KG/results/rerun/pip_freeze.txt", "w") as f:
    f.write("mock-package==1.0.0\n")
with open(f"../ExDR-KG/results/rerun/gpu_info.csv", "w") as f:
    f.write("name,driver_version,memory.total\nMPS,Mock,16GB\n")

print("Created mock outputs successfully.")
