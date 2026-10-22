# -*- coding: utf-8 -*-
"""
Created on Tue May 19 18:57:58 2026

@author: WangX3
"""

import re
from pathlib import Path

# input/output files
input_file = Path("Outer.dep")
output_file = Path("Outer_reverted.dep")

# Delft nodata marker
NODATA = -999.0

# read file
text = input_file.read_text()

# regex to match integers/floats/scientific notation
pattern = re.compile(r'[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?')

def flip_sign(match):
    s = match.group(0)
    v = float(s)

    # keep nodata unchanged
    if abs(v - NODATA) < 1e-9:
        return s

    flipped = -v

    # preserve decimal precision
    if '.' in s:
        decimals = len(s.split('.')[-1].split('E')[0].split('e')[0])
        return f"{flipped:.{decimals}f}"
    else:
        return str(int(flipped))

# apply conversion
new_text = pattern.sub(flip_sign, text)

# write output
output_file.write_text(new_text)

print(f"Saved reverted file to: {output_file}")