# efcamdat-multitask-learner-profiling

A multi-task Transformer model that predicts **CEFR proficiency level**, **L1 (native language)**, and **nationality** from learner English essays — with token-level SHAP explanations for each prediction head.

Built on RoBERTa and trained on the [EFCAMDAT](https://ef.com/assethub/cambridge) corpus of learner writing, using both the original (erroneous) text and its teacher-corrected version as a dual-input signal.

---

## What it does

| Task | Input | Output |
|------|-------|--------|
| CEFR classification | Learner essay | A1 → C1 proficiency level |
| L1 identification | Learner essay | Native language (Arabic, French, German, Italian, Japanese, Mandarin, Portuguese, Russian, Spanish, Turkish) |
| Nationality prediction | Learner essay | Learner nationality code |

All three tasks are learned simultaneously via a **shared RoBERTa encoder** with three independent classification heads — one model, not three.

---

## Architecture

```
"[RAW] I have 24 years [CORRECTED] I am 24 years old"
                        │
              RoBERTa encoder (roberta-base)
              + Dropout(0.1) on [CLS]
                        │
                [CLS] representation
               /        |        \
        CEFR head    L1 head    Nat head
            │            │          │
          A1–C1       Portuguese   BR
```

The dual-text format (`[RAW] ... [CORRECTED] ...`) lets the model learn from **error patterns**, not just content. Special tokens `[RAW]` and `[CORRECTED]` are added to the tokenizer vocabulary. Loss is the sum of three class-weighted cross-entropy losses, one per head, backpropagated jointly.

---

## Dataset

[EFCAMDAT (EF Cambridge Open Language Database)](https://ef.com/assethub/cambridge) — ~317,000 learner essays across CEFR levels A1–C1, with L1 and nationality metadata. Not included in this repo; access via the EF/Cambridge research portal.

### Class distribution

![Class distribution](images/class_distribution.png)

The corpus is heavily skewed at every level. CEFR is dominated by beginner essays — A1 alone accounts for ~47% of all samples. L1 and nationality are dominated by Portuguese/Brazilian learners at ~51%. This imbalance is addressed through class-weighted loss applied to all three heads, forcing the model to penalise minority-class errors more heavily during training.

### Word count by CEFR level

![Word count by CEFR](images/wordcount_by_cefr.png)

Average essay length scales almost linearly with CEFR level — from ~40 words at A1 to ~170 words at C1. This is a strong and expected signal: more proficient learners produce longer, more sustained writing. The model likely exploits this through the tokeniser's sequence length, meaning CEFR classification may partly be solved by essay length alone before any grammatical analysis. The planned raw-text-only ablation will help isolate how much is length vs. linguistic quality.

---

## Results

Trained for 3 epochs on an RTX 5060 (bf16 mixed precision, batch size 64, cosine LR schedule with warmup). Dataset: 317,220 essays split 80/20 (train/test, seed 42).

### Training curves

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1 | 3.0559 | 2.2516 |
| 2 | 1.8972 | 1.9192 |
| 3 | 1.4213 | 1.8728 |

Both losses decrease consistently across all three epochs — no divergence or overfitting observed within this run.

---

### CEFR classification — Accuracy: 98.1%

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| A1 | 0.99 | 0.99 | 0.99 | 29,896 |
| A2 | 0.99 | 0.98 | 0.98 | 17,189 |
| B1 | 0.98 | 0.96 | 0.97 | 10,928 |
| B2 | 0.93 | 0.96 | 0.94 | 4,443 |
| C1 | 0.88 | 0.95 | 0.91 | 988 |
| **weighted avg** | **0.98** | **0.98** | **0.98** | 63,444 |

![CEFR confusion matrix](images/cefr_confusion_matrix.png)

The confusion matrix tells a more nuanced story than the headline accuracy. Errors are almost entirely **adjacent-band** — B2 essays predicted as B1, C1 essays predicted as B2 — which mirrors how human raters also struggle at band boundaries. Notably, the model never predicts C1 as A1 or A1 as C1: extreme misclassification is essentially zero. The B1/B2 boundary (235 misclassifications each way) is the weakest point, consistent with the SLA literature where intermediate plateau learners are hardest to distinguish by surface error patterns alone. C1 recall of 95% from fewer than 1,000 test examples is the strongest indicator that class-weighted loss worked as intended.

---

### L1 identification — Accuracy: 69.7%

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Arabic | 0.61 | 0.75 | 0.67 | 3,280 |
| French | 0.51 | 0.64 | 0.57 | 2,704 |
| German | 0.56 | 0.76 | 0.65 | 3,322 |
| Italian | 0.43 | 0.69 | 0.53 | 3,294 |
| Japanese | 0.53 | 0.76 | 0.62 | 1,596 |
| Mandarin | 0.77 | 0.81 | 0.79 | 6,333 |
| Portuguese | 0.95 | 0.66 | 0.78 | 32,671 |
| Russian | 0.55 | 0.74 | 0.63 | 3,383 |
| Spanish | 0.51 | 0.72 | 0.60 | 6,087 |
| Turkish | 0.34 | 0.64 | 0.45 | 774 |
| **macro avg** | **0.58** | **0.72** | **0.63** | 63,444 |

![L1 confusion matrix](images/l1_confusion_matrix.png)

The L1 confusion matrix reveals linguistically meaningful error patterns. The largest off-diagonal confusions are between languages that share typological features or transfer error profiles: Italian and Spanish are frequently confused with each other and with Portuguese — all three are Romance languages whose speakers make similar determiner, copula, and preposition errors in English. Russian and Arabic show moderate confusion, both being non-Romance languages with no articles in their L1, leading to characteristic article-omission errors that may overlap in the model's feature space. Portuguese's high precision (0.95) but lower recall (0.66) reflects the class imbalance — the model is conservative about predicting Portuguese unless very confident. Turkish at macro F1 0.45 is the weakest class, unsurprisingly given its tiny support, but the 64% recall demonstrates the model has learned some signal from agglutinative transfer patterns.

---

### Nationality prediction — Accuracy: 68.2%

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| br | 0.95 | 0.65 | 0.78 | 32,671 |
| cn | 0.71 | 0.73 | 0.72 | 4,162 |
| de | 0.57 | 0.76 | 0.65 | 3,322 |
| fr | 0.50 | 0.64 | 0.56 | 2,704 |
| it | 0.42 | 0.69 | 0.52 | 3,294 |
| jp | 0.56 | 0.74 | 0.64 | 1,596 |
| mx | 0.51 | 0.72 | 0.60 | 6,087 |
| ru | 0.56 | 0.74 | 0.64 | 3,383 |
| sa | 0.62 | 0.74 | 0.67 | 3,280 |
| tr | 0.34 | 0.64 | 0.45 | 774 |
| tw | 0.45 | 0.61 | 0.52 | 2,171 |
| **macro avg** | **0.56** | **0.70** | **0.61** | 63,444 |

![Nationality confusion matrix](images/nationality_confusion_matrix.png)

Nationality performance mirrors L1 closely — the two tasks are near-perfectly correlated in EFCAMDAT (br ≈ Portuguese, cn/tw ≈ Mandarin, de ≈ German). The tw/cn split is the most interesting case: Taiwanese (tw) and Mainland Chinese (cn) learners share the same L1 but are treated as separate nationality classes. The confusion between them (541 tw predicted as cn) is higher than any other cross-nationality confusion, which makes intuitive sense — the model has no genuine signal to distinguish them beyond subtle stylistic variation, since the underlying transfer patterns from Mandarin are the same.

---

### Latent embedding space (CLS vectors)

**t-SNE** (non-linear, preserves local structure):

![t-SNE embeddings](images/tsne_embeddings.png)

**PCA** (linear, preserves global variance):

![PCA embeddings](images/pca_embeddings.png)

The embedding visualisations are among the most interpretable outputs of the project. **CEFR structure is strongly linear** — in the PCA plot, CEFR levels form a near-continuous gradient from top to bottom, confirming that the model has learned a meaningful proficiency axis in its representation space. The linearity is significant: it suggests proficiency is encoded as a genuine dimension in the encoder's geometry, not merely a lookup table. **L1 clusters are visible but overlapping** in t-SNE, with clear separation for Japanese and Mandarin but heavy overlap for the Romance languages — consistent with their shared typological features. The fact that L1 does not cleanly separate in PCA confirms the task requires non-linear features, which is why L1 accuracy is considerably harder than CEFR.

---

## SHAP — what the model learned

Token-level SHAP values were computed separately for the RAW and CORRECTED segments across three essays (A2, B2, C1), covering all three prediction heads. Red bars = tokens pushing toward the predicted class; blue bars = tokens pushing against it.

---

### Essay 1 — A2 | Portuguese | Brazil

> *"This Monday, there is going to be a play in Ibirapuera Park. The play starts at 9 am. Admission is free."*

**CEFR head (A2):**

![SHAP A2 CEFR](images/shap_a2_cefr.png)

"Monday" dominates both RAW and CORRECTED with a SHAP value of +0.65 — by far the single largest signal in any A-level essay. Simple temporal anchors, short declarative sentences, and basic vocabulary ("free", "9 am", "Admission is") are the defining A2 features. The "." period is slightly negative — punctuation alone carries no proficiency signal. This is consistent with ELT frameworks: A2 essays are characterised by short, formulaic sentences around concrete topics rather than complexity.

**L1 head (Portuguese):**

![SHAP A2 L1](images/shap_a2_l1.png)

The dominant tokens are "era", "ap", "u" — subword fragments of **Ibirapuera**, a famous park in São Paulo. The model has learned that this Brazilian proper noun is a direct signal for Portuguese L1. "Admission" and "is" are negative (pushing away from Portuguese), while the place name fragments dominate. This is a critical finding: the model is partly exploiting **topical and cultural references** rather than purely linguistic transfer features. A learner writing about a São Paulo landmark is almost certainly Brazilian regardless of grammar.

**Nationality head (br):**

![SHAP A2 Nationality](images/shap_a2_nationality.png)

Identical pattern — "Ibirapuera" fragments ("era", "ap", "u") completely dominate. The nationality head and L1 head are essentially solving the same problem the same way here, using geography as a proxy for identity. This is a legitimate shortcut the model has found, but it is not transfer-error-based reasoning — a point worth noting for any downstream application.

---

### Essay 2 — B2 | French | France

> *"The French like to honour their good food with good presentation and good manners... At the table in France, manners are important even while eating with your family..."*

**CEFR head (B2):**

![SHAP B2 CEFR](images/shap_b2_cefr.png)

Strikingly different from the A2 pattern. No single token dominates — all shown tokens ("manners", "are", "important", "even", "while", "eating", "with", "your", "family") contribute almost uniformly at around 0.12–0.14. The CORRECTED panel shows the same even distribution across "your hands on the table", "Here are some helpful tips". The model is reading **register and sustained coherence** rather than any individual word. This is pedagogically meaningful — B2 is precisely where learners develop the ability to produce extended, organised discourse on social/cultural topics.

**L1 head (French):**

![SHAP B2 L1](images/shap_b2_l1.png)

A striking finding: the RAW segment is dominated by "France" (+0.5) and "French" (+0.35). The CORRECTED segment is **entirely blue** — every token pushes *away* from French L1 prediction. The model's L1 signal for French in this essay comes almost entirely from the explicit mention of the country and people in the text content, not from any grammatical transfer feature. When the corrected text (which doesn't happen to contain these nouns in the top tokens) is processed, the model loses confidence. This is a significant limitation: the model conflates writing *about* France with writing *as* a French speaker.

**Nationality head (fr):**

![SHAP B2 Nationality](images/shap_b2_nationality.png)

The same pattern as L1 — "France" and "French" dominate the RAW signal, while the CORRECTED panel is uniformly negative. The corrected fragment "Drinks won't be served until the last guests arrive" actively pushes *away* from French nationality prediction, which is unexpected. This further confirms the model is not detecting French transfer errors here but is instead reading topic keywords as nationality proxies.

---

### Essay 3 — C1 | German | Germany

> *"The Euro is approximately ten years old... For ordinary citizen it was quite difficult to make out the benefits of the change... Even today the German citizen do not have a lot of confidence in the new currency..."*

**CEFR head (C1):**

![SHAP C1 CEFR](images/shap_c1_cefr.png)

Again a uniform distribution, but now the phrases are qualitatively different from B2. "For ordinary citizen it was quite difficult to make out the benefits" and "We all know that the introduction of the Euro right and" — these are complex nominal constructions and long subordinate clauses. The absence of any single dominant token confirms the model is reading **sustained syntactic complexity** as the C1 signal. Interestingly the essay has genuine errors ("goverment", "decates", "inportant") which do not appear to suppress the C1 prediction — the model correctly weights the overall complexity over the surface spelling errors.

**L1 head (German):**

![SHAP C1 L1](images/shap_c1_l1.png)

The top signals in the CORRECTED panel are "Even today the **German** **citizen** do not have" — with "German" and "citizen" as the highest-weight tokens (+0.2). This is genuinely ambiguous: "German citizen" is both a topical reference (writing about Germans) and a site of a real transfer error — "the German citizen **do** not have" is a subject-verb agreement error characteristic of German L1 speakers (German uses different plural agreement rules). The model may be detecting the error, the content reference, or both. The phrase "the German citizen do not have" is notable: it's ungrammatical in English but follows German grammatical logic where collective nouns can take different agreement. This is precisely the kind of transfer error the SLA literature predicts for German L1 learners of English.

**Nationality head (de):**

![SHAP C1 Nationality](images/shap_c1_nationality.png)

Near-identical to the L1 German plot. "Even today the German citizen do not have" and "We all know that" dominate. The convergence of the L1 and nationality heads on the same tokens confirms these two heads have learned near-identical representations for this class, which is expected given their near-perfect correlation in the dataset.

---

### Cross-essay SHAP observations

**1. CEFR decoding strategy shifts with level.** At A2, single concrete words dominate (sparse signal). At B2 and C1, the signal is distributed uniformly across many tokens (dense signal). This mirrors the cognitive shift in proficiency: beginners write around keywords, advanced learners produce integrated discourse where no single word carries the meaning alone.

**2. L1 prediction is partly topical, not purely transfer-based.** Two of three essays show the model exploiting cultural/geographic content (Ibirapuera for Portuguese, France/French for French, German/Germany for German) as proxies for L1. This is a meaningful limitation for any application where the essay topic is controlled — a French learner writing about a neutral topic would receive weaker L1 signal from this model.

**3. The CORRECTED segment behaves differently across levels.** At A2 the corrected and raw panels are nearly symmetric. At B2 the corrected panel actively suppresses French L1 prediction. At C1 the corrected panel carries strong positive German signal from "German citizen do not have." The dual-text design is not uniformly beneficial — its effect on L1 prediction is highly essay-dependent.

**4. Genuine transfer errors are detectable but not dominant.** The "German citizen do not have" agreement error is the clearest example of the model responding to a real transfer pattern from the SLA literature. Finding more such cases systematically would require running SHAP across a larger, topic-controlled sample — a clear direction for future work.

---

## Project structure

```
efcamdat-multitask-learner-profiling/
├── Cambridge_Models_final.ipynb   # Main notebook (13 cells: train, eval, SHAP)
├── images/                        # All evaluation plots and SHAP visualisations
├── requirements.txt
└── README.md
```

---

## Requirements

```
transformers>=4.40
datasets
torch>=2.0
scikit-learn
shap
pandas
tqdm
matplotlib
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Prepare data

| Column | Description |
|--------|-------------|
| `text` | Raw learner essay |
| `text_corrected` | Teacher-corrected version |
| `cefr` | CEFR label string (A1, A2, …) |
| `l1` | Native language string |
| `nationality` | Nationality code |

### 2. Run the notebook

Works on Colab (mount Drive, set `DATA_DIR`) or locally (place CSV in the same folder as the notebook). GPU recommended — training on an RTX 5060 takes ~50 minutes per epoch at batch size 64 with bf16.

### 3. Inference on a new essay

```python
essay_text     = "I have 24 years and I work in a company."
corrected_text = "I am 24 years old and I work for a company."

inputs = tokenizer(
    ["[RAW] " + essay_text + " [CORRECTED] " + corrected_text],
    truncation=True, padding='max_length', max_length=128, return_tensors='pt'
)

multitask_model.eval()
with torch.no_grad():
    outputs = multitask_model(**{k: v.to(device) for k, v in inputs.items()})

print(cefr_encoder.inverse_transform([outputs['cefr'].argmax().item()]))
print(l1_encoder.inverse_transform([outputs['l1'].argmax().item()]))
print(nat_encoder.inverse_transform([outputs['nat'].argmax().item()]))
```

### 4. Explain a prediction with SHAP

```python
explainer = shap.Explainer(make_shap_predictor(multitask_model, 'cefr', device), tokenizer)
shap_values = explainer(["[RAW] I have 24 years [CORRECTED] I am 24 years old"])
shap.plots.text(shap_values[0])
```

---

## Planned work

- Topic-controlled SHAP analysis on essays with neutral subject matter to isolate genuine transfer error signal from topical content leakage
- Raw-text-only ablation to quantify the dual-text contribution to CEFR accuracy
- Balanced undersample run (capped per-class) for direct comparison against this baseline
- Extend to 5 epochs with early stopping now that 3-epoch loss curves confirm stable training
- Wrap in a FastAPI + React app: upload essay → predictions + SHAP highlight
- Cross-corpus evaluation on CasiMedicos or NUCLE to test generalisability

---

## Author

Darragh — MSc Language Analysis and Processing, UPV/EHU | CELTA-certified ELT practitioner  
[GitHub: VerbalAid](https://github.com/VerbalAid)
