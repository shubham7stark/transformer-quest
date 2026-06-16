#!/usr/bin/env python3
"""Inline the modular sources into the single self-contained ../index.html.

Reads shell.html and replaces three markers:
  /*__STYLES__*/  <- styles.css
  /*__KERNEL__*/  <- kernel.js
  /*__LEVELS__*/  <- levels/*.js concatenated in filename order (01..10)

Usage:  python3 build/build.py     (run from the repo root)
"""
import pathlib

HERE = pathlib.Path(__file__).resolve().parent          # build/
ROOT = HERE.parent                                       # repo root

shell  = (HERE / "shell.html").read_text(encoding="utf-8")
styles = (HERE / "styles.css").read_text(encoding="utf-8")
kernel = (HERE / "kernel.js").read_text(encoding="utf-8")

level_files = sorted((HERE / "levels").glob("*.js"))
levels = "\n\n".join(
    f"/* ===== {p.name} ===== */\n" + p.read_text(encoding="utf-8")
    for p in level_files
)

for marker, payload in (
    ("/*__STYLES__*/", styles),
    ("/*__KERNEL__*/", kernel),
    ("/*__LEVELS__*/", levels),
):
    if marker not in shell:
        raise SystemExit(f"marker {marker} not found in shell.html")
    shell = shell.replace(marker, payload)

out = ROOT / "index.html"
out.write_text(shell, encoding="utf-8")
print(f"wrote {out}  ({len(shell)//1024} KB, {len(level_files)} levels)")
