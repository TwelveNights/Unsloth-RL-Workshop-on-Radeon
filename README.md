Provides a Jupyter Notebook that runs on AMD Radeon machines to fine-tune a model to play Sudoku using GRPO and unsloth.
The attached presentation explains the why and how of fine-tuning. The notebook contains code for Sudoku too.
The learning parameters were adjusted to complete a training run in 20-30 minutes,
for a workshop-length presentation.

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

## Unsloth Studio track (no-code)

A second workshop on the same concepts, where attendees fine-tune in
[Unsloth Studio](https://github.com/unslothai/unsloth), Unsloth's web UI, instead
of driving training code. Studio's web UI trains with supervised fine-tuning
(LoRA, QLoRA, full fine-tune, continued pretraining) and has no GRPO / reward
functions, so the Sudoku task is replaced with one SFT suits: turning a messy
customer support message into a ticket with a fixed JSON schema. Attendees
train `unsloth/Qwen3.5-2B` in Studio (60 steps, about a minute on an MI300X),
compare base and fine-tuned side by side in Studio's Chat, then score both on
held-out tickets in the notebook.

| File | Use |
|---|---|
| `Unsloth_Studio_Support_Tickets.ipynb` | The hands-on notebook: data, step-by-step Studio walkthrough with screenshots, scoring, export. |
| `docs/Unsloth_Studio_Support_Tickets-executed-mi300x.ipynb` | The same notebook run end to end on an MI300X against a Studio-trained adapter: fine-tuned 100% exact on 30 held-out tickets, versus 70% for the base model with a full schema prompt. |
| `docs/AMD-Unsloth-Studio-Fine-tuning.pdf` / `.pptx` | The deck: the GRPO deck's concepts, with Part 2 rebuilt around Studio. Same PDF/PPTX rule as above. |
| `docs/studio-screenshots/` | Studio screenshots used by the notebook and deck. |
| `data/support_tickets_{train,eval}.jsonl` | 1,000 training and 100 held-out tickets, chat (`messages`) format. |
| `scripts/generate_support_ticket_dataset.py` | Regenerates the data deterministically. |
| `scripts/score_tickets.py` | Base vs schema-prompted vs fine-tuned scoring, shared with the notebook. |

Facilitator notes, from testing on Studio 2026.10.4 / ROCm:

- **Why Qwen3.5-2B, not Gemma 4.** Studio strips the leading `<bos>` from Gemma
  chat datasets before training and the tokenizer does not add it back, while
  Studio's Chat and Transformers serve the model with `<bos>`. The Gemma 4
  adapter writes the right fields and then keeps inventing keys. Qwen's chat
  template has no `<bos>`, so training and serving match. Re-test Gemma 4 after
  a Studio upgrade before switching back.
- **Chat format, not Alpaca.** Studio trains Alpaca rows with a generic
  `### Instruction / ### Response` prompt rather than the model's chat template,
  so an Alpaca-trained model is tested in Chat on a prompt it never saw.
- **First training run compiles a kernel.** Qwen3.5 needs `causal-conv1d`, which
  Studio builds from source for ROCm on first use (several minutes). Do one run
  on each machine image before the event.
- **Training fails at step 0 with `'KernelMetadata' object has no attribute
  'cluster_dims'`.** Studio's installer put upstream `triton` over
  `pytorch-triton-rocm` on a ROCm 6.4 host. Uninstall `triton` and
  force-reinstall the `pytorch-triton-rocm` that matches the installed torch,
  in Studio's venv.
- **Thinking off in Chat.** With Thinking on, the prompt no longer matches the
  training examples and the fine-tuned model drifts.
