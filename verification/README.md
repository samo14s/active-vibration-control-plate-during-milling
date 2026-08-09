# Verification scripts

Reproduce every number in `../VERIFICATION.md`. Run from the repository root:

```bash
pip install numpy scipy openpyxl
python3 verification/01_frf_vs_measured.py
```

Each script prepends `simulation/sim_kit` to `sys.path` and chdir's there
(the digitized `.xlsx` curves live next to the model), so the working
directory you launch from does not matter.
