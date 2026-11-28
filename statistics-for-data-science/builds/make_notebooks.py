#!/usr/bin/env python3
"""Make the build notebooks (.ipynb) from the build scripts (`# %%` cells, jupytext percent format).

The `.py` files stay the source. Each notebook gets one extra first cell that only acts in Google Colab: it clones
this repository, moves into the course folder, installs the few packages Colab lacks for that build, generates the
build's data (once per session) and moves into builds/. On your own machine that cell does nothing.

Run from the course folder after editing a build:  python builds/make_notebooks.py
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = "https://github.com/CoralPetalAI/CoralPetalAI-Academy.git"
# datasets each build loads (datasets/generate_petalmart_<name>.py writes datasets/data/petalmart_<name>/)
DATA = {"00": ["core"], "01": ["core"], "02": ["core"], "03": ["customers"], "04": ["core"], "05": ["core"],
        "06": ["city_conversion"], "07": ["weekly", "customers"], "08": ["repeat"], "09": ["demand"],
        "10": ["checkout_test"], "11": ["same_day"], "12": []}
# packages a build imports that are not in Colab's standard runtime
EXTRA = {"01": ["duckdb"], "06": ["pymc", "arviz"]}

SETUP = '''# Google Colab only: get the course code and this build's data. On your own machine this cell does nothing.
import os, subprocess, sys
if "google.colab" in sys.modules:
    if not os.path.isdir("/content/CoralPetalAI-Academy"):
        subprocess.run(["git", "clone", "-q", "{repo}", "/content/CoralPetalAI-Academy"], check=True)
    os.chdir("/content/CoralPetalAI-Academy/statistics-for-data-science")
{body}    os.chdir("builds")
    print("Ready: the course code is cloned and the data is generated.")'''


def setup_cell(num):
    body = ""
    if EXTRA.get(num):
        body += f'    subprocess.run([sys.executable, "-m", "pip", "install", "-q", {", ".join(json.dumps(p) for p in EXTRA[num])}], check=True)\n'
    for name in DATA[num]:
        body += (f'    if not os.path.isdir("datasets/data/petalmart_{name}"):\n'
                 f'        subprocess.run([sys.executable, "datasets/generate_petalmart_{name}.py"], check=True)\n')
    return SETUP.format(repo=REPO, body=body)


def cells_from_percent(text):
    """Split a jupytext percent-format script into (kind, source) cells."""
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        lines = buf[:]
        while lines and not lines[-1].strip():
            lines.pop()
        if kind == "markdown":
            lines = [re.sub(r"^# ?", "", ln) for ln in lines]
        cells.append((kind, "\n".join(lines)))

    for line in text.splitlines():
        m = re.match(r"^# %%(.*)$", line)
        if m:
            flush()
            kind, buf = ("markdown" if "[markdown]" in m.group(1) else "code"), []
        elif kind is not None:
            buf.append(line)
    flush()
    return cells


def notebook(cells):
    out = []
    for i, (kind, src) in enumerate(cells):
        lines = src.split("\n")
        source = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]
        c = {"cell_type": kind, "id": f"cell-{i:02d}", "metadata": {}, "source": source}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        out.append(c)
    return {"cells": out, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def main():
    for py in sorted(HERE.glob("build_*.py")):
        num = py.stem.split("_")[1]
        cells = cells_from_percent(py.read_text(encoding="utf-8"))
        # keep the title cell first, then the Colab cell, then the build
        cells = cells[:1] + [("code", setup_cell(num))] + cells[1:]
        nb = py.with_suffix(".ipynb")
        nb.write_text(json.dumps(notebook(cells), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(nb.name, len(cells), "cells")


if __name__ == "__main__":
    main()
