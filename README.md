Provides a Jupyter Notebook that runs on AMD Radeon machines to fine-tune a model to play Sudoku using GRPO and unsloth.
The attached presentation explains the why and how of fine-tuning. The notebook contains code for Sudoku too.
The learning parameters were adjusted to complete a training run in 20-30 minutes,
for a workshop-length presentation.

## Bonus: SFT notebook

`Gemma4_(E2B)_SFT_Sudoku_Board_Parsing.ipynb` is a short (about 10 minute) extra for
people who finish early. It fine-tunes the same Gemma 4 E2B with supervised fine-tuning
to turn an 81-character Sudoku string into the `board` list the GRPO notebook uses, and
scores the model on held-out puzzles before and after. Its dataset is in `data/`
(1,000 train and 50 test examples, under 1 MB); `python data/make_sudoku_sft.py`
regenerates it.

## Slides

The deck for the talk is 19 slides on the AMD corporate template, with a notebook
access slide up front, two section dividers ("Part 1: The Concepts" and "Part 2:
Hands-On") separating the theory from the hands-on walkthrough, and full-bleed AMD
background imagery.

| File | Use |
|---|---|
| `docs/AMD-Unsloth-GRPO-Fine-tuning1.pdf` | Presenting. Renders identically everywhere, with no dependency on PowerPoint or on the AMD corporate fonts. |
| `docs/AMD-Unsloth-GRPO-Fine-tuning1.pptx` | Editing. The source the PDF is exported from. |

If you edit the `.pptx`, re-export the PDF in the same commit so the two stay in sync.
