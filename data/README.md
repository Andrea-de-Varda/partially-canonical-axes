# Data

The MIT license of this repository covers the code only. The files below come from third parties and remain under their own terms. **If you use any of them, cite the original paper.**

| File | Content | Source (cite) | License / terms |
|---|---|---|---|
| `language_data/blimp_data/*.jsonl` | 12 of the 67 BLiMP sub-tasks (one per phenomenon class), 1,000 pairs each, unmodified | Warstadt et al. (2020). BLiMP: The Benchmark of Linguistic Minimal Pairs for English. *TACL*. https://github.com/alexwarstadt/blimp | CC BY 4.0 |
| `language_data/wug_pairs.jsonl` | 1,000 Wug minimal pairs; the code drops the 2 pairs whose members are identical (998 used) | Hanna, Belinkov & Pezzelle (2026). Are formal and functional linguistic mechanisms dissociated in language models? *Computational Linguistics*. Files from https://github.com/hannamw/formal-functional-dissociation (`data/wug/`) | MIT, see `language_data/LICENSE-hanna` |
| `language_data/hypernymy_pairs.jsonl` | 1,000 hypernymy minimal pairs, containing 551 distinct well-formed sentences; the code keeps one pair per sentence (551 used) and prefixes "Consider" | Adapted from Hanna & Mareček (2021). Analyzing BERT's knowledge of hypernymy via prompting. *BlackboxNLP*. Files from the Hanna et al. (2026) repository (`data/hypernymy-comma/`) | MIT, see `language_data/LICENSE-hanna` |
| `perceptual.csv` | Lancaster Sensorimotor Norms (39,707 words), unmodified | Lynott, Connell, Brysbaert, Brand & Carney (2020). *Behavior Research Methods*. https://osf.io/7emr6/ | CC BY 4.0 |
| `affective.csv` (not included) | Valence, arousal, dominance for 13,915 lemmas; the analyses use the 13,372 words shared with the Lancaster norms | Warriner, Kuperman & Brysbaert (2013). *Behavior Research Methods* 45:1191–1207 | Not redistributed: `scripts/fetch_warriner.sh` downloads the article's supplementary file and checks its sha256 |
| `pereira_averaged_with_passage.csv` | The 627 sentences of Experiments 2 and 3 with their passage ids (used for passage-level cross-validation). The `EffectSize` column is a per-sentence response averaged over the language network; the code does not use it | Pereira et al. (2018). Toward a universal decoder of linguistic meaning from brain activation. *Nature Communications*. https://osf.io/crwz7 | Stimuli of a public dataset, reused with attribution |
| `pereira_language.npz` (download) | fMRI responses, see the main README ("Brain data") | Pereira et al. (2018); re-analysis by the Fedorenko lab | Cite Pereira et al. (2018) and this paper |
| `results/` | All per-model, per-layer result tables behind the paper (this work) | this paper | CC BY 4.0 |
