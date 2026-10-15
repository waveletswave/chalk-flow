# chalk-flow

Study notes and Python scripts for Appendix B, "Mathematical Tools and Resources", of Kundu, Cohen & Dowling, *Fluid Mechanics* (6th ed., 2016), with connections to Earth-surface dynamics.

## Progress

| Chapter | Topic                                                                         | Folder                        | Status      |
| ------- | ----------------------------------------------------------------------------- | ----------------------------- | ----------- |
| 1       | Partial and total derivatives, material derivative, method of characteristics | `ch01_derivatives/`         | In progress |
| 2       | Changing independent variables, moving frames, similarity variables           | `ch02_change_of_variables/` | In progress |
| 3       | Vector calculus and integral theorems                                         | `ch03_vector_calculus/`     | Not started |
| 4       | Dirac delta function and Green's functions                                    | `ch04_dirac_delta/`         | Not started |
| 5       | Three-dimensional coordinate systems                                          | `ch05_coordinates/`         | Not started |
| 6       | Equations in curvilinear coordinates                                          | `ch06_curvilinear/`         | Not started |
| 7       | Capstone: shoreline instability under high-angle waves                        | `ch07_capstone/`            | Not started |

## Repository layout

- `guide/`: the study guide (LaTeX source and PDF)
- `chXX_*/`: one folder per chapter, each containing
  - `chXX_notes.ipynb`: notes, derivations, code and figures
  - `chXX_tools.py`: reusable functions
  - `test_chXX.py`: automated checks against known results
  - `figures/`: figures worth showing on GitHub

## Getting started

1. Create the environment (first time only): `python3.12 -m venv .venv`, then `source .venv/bin/activate` and `pip install -r requirements.txt`
2. Set up notebook output stripping (once per clone): `nbstripout --install`
3. In later sessions, activate the environment: `source .venv/bin/activate`
4. Open the notebooks in VS Code and select the `.venv` kernel (or run `jupyter lab`)
6. Run the checks: `pytest`

## Reference

Kundu, P. K., Cohen, I. M., & Dowling, D. R. (2016). *Fluid Mechanics* (6th ed.). Academic Press.
The textbook PDF is not included in this repository.
