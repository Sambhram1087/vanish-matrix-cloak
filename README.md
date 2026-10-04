# Vanish: a matrix-powered invisibility cloak

Mini project, UE25MA242A Mathematical Foundation for AI & Data Science (MFAD 2026).

Video frames are stacked as columns of a matrix **M** and split into **M = L + S**
(low-rank background + sparse moving people) using SVD and Robust PCA. The person is
removed using a mask built with convolution and morphology.

## Setup
```bash
git clone <repo-url>
cd vanish-matrix-cloak
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q
python main.py
```

## Team
| Member | Name | GitHub | Module |
|---|---|---|---|
| 1 | | | Data + live pipeline (`frames.py`) |
| 2 | | | SVD + rank-k (`svd_tools.py`) |
| 3 | | | Robust PCA (`rpca.py`) |
| 4 | | | Masks + UI + integration (`masks.py`, `app.py`) |

See `CONTRIBUTING.md` for the workflow.
