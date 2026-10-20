# FB-01 environment — 2026-09-18

Windows AMD64; Python 3.12.10; Pygame 2.6.1; SDL 2.28.4.
Installed editable with `--only-binary=:all: -e ".[dev]"`.

Resolved development dependencies (`python -m pip freeze`):

```text
colorama==0.4.6
flybrain==0.1.0 (editable local package)
iniconfig==2.3.0
packaging==26.3
pluggy==1.6.0
pygame==2.6.1
Pygments==2.21.0
pytest==9.1.1
```

This records the checked environment, not a cross-platform guarantee. Python
3.14.2 had no compatible Pygame wheel; Python 3.12.10 was installed through the
approved user-scope winget command. The initial environment is retained in ignored
`.venv-py314/`; the working environment is `.venv/`.

## FB-03 neural environment — 2026-09-18

Same Windows AMD64 / Python 3.12.10 / Pygame 2.6.1 / SDL 2.28.4 / pytest 9.1.1.
CPU PyTorch installed from the official wheel index; editable package refreshed
with the optional `neural` extra (NumPy is the PyTorch interoperability prerequisite).

```powershell
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -e ".[dev,neural]"
```

Latest observed `python -m pip freeze`:

```text
colorama==0.4.6
filelock==3.32.3
flybrain==0.1.0 (editable local package)
fsspec==2026.7.0
iniconfig==2.3.0
Jinja2==3.1.6
MarkupSafe==3.0.3
mpmath==1.3.0
networkx==3.6.1
numpy==2.5.3
packaging==26.3
pluggy==1.6.0
pygame==2.6.1
Pygments==2.21.0
pytest==9.1.1
setuptools==78.1.0
sympy==1.14.0
torch==2.14.0+cpu
typing_extensions==4.16.0
```

`python -m pip check`: no broken requirements. CPU float32 and float64 checked;
no CUDA or cross-platform reproducibility claim. PyTorch initially warned about
missing NumPy; the installed extra supplies it. Full suite retains a Pygame
`pkg_resources` deprecation warning from setuptools. No GPU, vision/audio,
training or biological data package was installed.
Installation reference: [official PyTorch setup](https://docs.pytorch.org/get-started/locally/).

## FB-08 plotting extra (2026-09-23)

Standalone experiment plots use the optional `plots` extra:
`python -m pip install -e ".[plots]"`. The app and neural simulation do not
import Matplotlib. Installed with the workspace virtual environment:

```text
matplotlib==3.11.2
contourpy==1.4.0
cycler==0.12.1
fonttools==4.65.0
kiwisolver==1.5.1
pillow==12.3.0
pyparsing==3.3.3
python-dateutil==2.9.0.post0
six==1.17.0
```

Existing NumPy and neural package versions were retained. `python -m pip check`
reported no broken requirements. PNG/SVG figures use the noninteractive Agg
backend and are generated from saved JSON results.
