"""Run the full campaign and generate every manuscript figure."""
import subprocess
import sys
import pathlib

HERE = pathlib.Path(__file__).resolve().parent

STEPS = [
    "pipeline.py",
    "fig_model_validation.py",
    "fig_varying_dynamics.py",
    "fig_design_framework.py",
    "fig_sld.py",
    "fig_time_domain.py",
    "fig_c2.py",
    "fig_consistency.py",
    "fig_frf_match.py",
    "fig_robustness.py",
]

if __name__ == "__main__":
    for step in STEPS:
        print(f"===== {step} =====", flush=True)
        r = subprocess.run([sys.executable, str(HERE / step)])
        if r.returncode != 0:
            sys.exit(f"{step} failed with code {r.returncode}")
    print("all figures generated")
