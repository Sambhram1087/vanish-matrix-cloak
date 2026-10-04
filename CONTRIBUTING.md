# Team workflow (read this once)

| Member | Owns (only edit these) |
|---|---|
| 1 | `frames.py`, live mode (`live.py`) |
| 2 | `svd_tools.py` |
| 3 | `rpca.py` |
| 4 | `masks.py`, `app.py`, `main.py` (integration) |

## Daily routine
```bash
git checkout main
git pull                                  # 1. get everyone's latest work
git checkout member1-work                 # 2. your own branch (member1..member4)
git merge main                            #    bring main into your branch
# ... edit your file(s) ...
python -m pytest -q                       # 3. tests must pass
git add frames.py tests/
git commit -m "frames: implement load_video"
git push -u origin member1-work           # 4. push your branch
```
Then open a **Pull Request** on GitHub (`member1-work` -> `main`). Member 4 merges it after the tests are green.

## Rules
1. Never change a function's name, inputs or output shape (see the data contract in the team plan and `tests/test_contract.py`).
2. Only edit your own files. Need a change in someone else's file? Message them.
3. Commit small and often; pull before you start working.
4. Do not commit videos or big files (they are in `.gitignore`).
5. If `main` is red (tests failing), stop and fix it first.
