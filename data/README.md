# Data Directory Notes

## Purpose
- `data/real_test.txt` and `data/test_come.txt` are lightweight input corpora for pipeline smoke tests.
- `data/sample_uhvn.xml` is the frame/schema input used by `parse_uhvn.py`.
- `data/html_frames/` contains PropBank-style frame pages used to construct trigger vocabulary and role metadata.

## Preparation
- Keep all UTF-8 encoding.
- One sentence per line is expected for extraction inputs.

## Reproducibility checklist (to complete for final thesis)
- Record original corpus source and license for each public file.
- Note any manual cleaning performed before extraction.
- Record exact train/val/test splits (including random seeds and ratios).
- Commit the exact frame/model versions used to produce each artifact.
