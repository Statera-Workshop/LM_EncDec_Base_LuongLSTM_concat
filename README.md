# LM_EncDec_Base_LuongLSTM_concat

Encoder-decoder translation model ported from legacy TF2 code C5 (concat) to PyTorch (cell=lstm, attention=concat).

## Status

- Verification level: **Benchmarked (LM-03, one run)**

## Changes from the legacy code

- PyTorch port; beam search reimplemented (TF Addons is no longer maintained)
- Vocabulary built from the training split only (legacy built it from all pairs)
- "what's" expanded to "what is" (legacy wrote "that is")

## LM-03 setting

English -> French, 5-11 words, 12,800 pairs (seed 1234), 80/20 split, embedding 128, hidden 1024, batch 256, 20 epochs, Adam 1e-3. Small data: compare structures, not absolute BLEU.

## Results

- Parameters: 27.2 M
- Test perplexity 31.7, next-word accuracy 0.567
- BLEU-4 greedy (whole test split): 21.1
- BLEU-4 beam width 4 on 500 pairs: 22.9 (greedy on the same pairs 22.3)
- Examples:
  - `this winter has been mild .` -> `cette femme est venu de cette cuisine .` (reference `cet hiver a ete doux .`)
  - `they will be here at three .` -> `ils sera ici a trois .` (reference `ils seront ici a trois heures .`)
  - `i once lived in rome .` -> `j'ai une autre detonation et je parle .` (reference `j'ai vecu un temps a rome .`)

## Run

```bash
pip install torch==2.5.1 einops numpy
# put fra-eng.zip from https://www.manythings.org/anki/ into data/
python run.py lm03 s2s_luong_lstm_concat
```
