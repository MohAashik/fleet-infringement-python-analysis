# Fleet Infringement Cost Leakage & Recovery Triage

**Where a rental fleet actually loses money on traffic fines and tolls, why, and which notices to work first.**

Python · pandas · scikit-learn · matplotlib

---

## About this project

I process fines, tolls and infringement notices for a fleet of 120+ Australian vehicles as part of my day job, including direct liaison with Revenue NSW.

That data is confidential and does not appear here. Instead I rebuilt the workflow from scratch — the statutory nomination window, the liability transfer, the toll rebill mechanics — as a data generator, then analysed the output as if it were a register I had been handed.

**The domain knowledge is real. The data is synthetic.** Both of those statements matter, and the limitations section below says exactly what that means for the conclusions.

---

## How liability actually works

This is the part most analyses of this problem get wrong. **Fines and tolls behave in opposite ways.**

**Infringements** (speed, red light, parking, bus lane, mobile phone) arrive in the company's name, because the company is the registered owner. There is then a statutory window — 21 days — to nominate the hirer who was driving.

Nominate in time and **liability transfers completely**. The penalty moves into the hirer's name and the issuing authority pursues them directly. It makes no difference whether that person has since flown home, ignores email, or left no card on file — that is now between them and Revenue NSW. The company's exposure is zero.

Miss the window and the nomination is rejected, and **the company pays the fine itself, permanently, with no appeal.**

**Tolls** work the other way. There is no nomination process. The toll operator bills the registered owner, so the company pays first and rebills the hirer afterwards. Here the money genuinely does have to be collected, and a hirer who has gone overseas with no card on file is a real, permanent loss.

So the company absorbs cost in exactly three ways:

| Root cause | Owner | Controllable? |
|---|---|---|
| Missed nomination deadline | Back office | Fully — it is a capacity problem |
| No hire found for the notice | Fleet ops | Mostly — movement and disposal records |
| Billable never collected | Front counter / billing | Partly — data capture at pickup |

---

## Headline results

Over 25 months, 7,734 clean notices:

| | |
|---|---|
| Face value handled | $1,169,708 |
| **Penalties transferred to hirers** | **$934,084** (cost to company: $0) |
| Penalties absorbed | $188,034 |
| Billable raised on hirers | $162,138 |
| Billable never collected | $28,815 |
| **Total cost to business** | **$216,849** — 18.5% of face value, ~$8,674/month |

The process mostly works. Most penalty value is successfully transferred. The cost sits in the tail, and the point of the analysis is to size that tail and attribute it to an owner.

### Cost by root cause

| Cause | Cost | Share | Notices | Avg per notice |
|---|---|---|---|---|
| Missed nomination deadline | $123,866 | 57.1% | 322 | **$385** |
| No hire found | $64,168 | 29.6% | 415 | $155 |
| Billable not collected | $28,815 | 13.3% | 1,323 | **$22** |

A missed deadline costs roughly **100x** an uncollected rebill. That asymmetry drives everything downstream.

---

## Findings

**1. Cost is extremely concentrated.** About a quarter of notices generate some cost, but just 737 of them (9.5% of all notices) account for 87% of it. That is the argument for triage over first-in-first-out processing.

**2. Liability transfer is binary.** A fine nominated inside 21 days costs nothing regardless of whether the hirer is contactable. One day late costs the full amount, permanently. There is no middle ground, which is what makes the deadline worth protecting above everything else.

**3. Cost tracks the back-office backlog, not notice volume.** Correlation between monthly cost and missed-deadline rate: **+0.87**. Between cost and notice volume: **+0.07**. The worst months had unremarkable volume and average processing times pressed up against the 21-day deadline.

**4. The notices that fill the queue are not the notices that cost money.** Tolls are 58% of volume and a small fraction of cost — each is worth a few dollars and rebills automatically. Camera-detected fines are the reverse.

**5. A missing card on file is the strongest predictor of an uncollected billable** — and it is set at the counter, months before any notice arrives.

| Collection failure rate | Domestic | Overseas |
|---|---|---|
| Card on file | 11.9% | 24.3% |
| No card on file | 29.9% | 43.5% |

**6. 95% of the recoverable opportunity comes from process fixes, not from chasing customers harder.** Sizing the three interventions (with stated, arguable assumptions): clearing the nomination backlog $99,093, daily fleet-movement reconciliation $38,501, mandating a card at pickup $7,641 — $145,235 over 25 months, ~$69,713 annualised.

---

## Modelling

Cost arrives through two unrelated failures, so this is two prediction problems, not one. An earlier version trained a single model on everything and scored barely above guessing.

| | Predicts | Driven by | Test AUC |
|---|---|---|---|
| **Model 1 — deadline risk** | Will this fine miss the 21-day window? | Team capacity, seasonality | 0.553 |
| **Model 2 — collection risk** | Will this billable ever be collected? | Hirer payment and contact details | **0.681** |

**Logistic regression** for both. The target is binary and the output needs to be a 0–1 probability for a triage score. `class_weight="balanced"` is essential — only ~19% fail, so without it the model predicts "will be fine" for everything, looks 81% accurate, and is useless. Linear regression would be the wrong tool and is not used: it predicts a quantity and can output -0.3 or 1.8 for a yes/no question.

**Gradient boosting was tried and lost** (0.647 vs 0.681). Permutation importance shows why: `card_on_file` at 0.089 with every other feature below 0.02. One dominant flag plus weak roughly-additive extras is the shape logistic regression handles best. Boosted trees need complex interactions to earn their flexibility; without them, the extra capacity just fit the training year's noise.

Both are then combined through the domain rules rather than assuming one model can learn them:

```
expected cost = P(miss deadline) × penalty
              + (1 − P(miss deadline)) × P(no collect) × billable
```

For a toll, `P(miss deadline)` is zero by definition — there is no deadline — so it reduces to collection risk alone.

### Method

- **Time-based split** — train on 2024, test on 2025+. A random split lets the model learn from the future and flatters the score.
- **12 post-outcome columns excluded, and the exclusion is `assert`ed in code**, not merely intended.
- **Preprocessing inside a `Pipeline`** so imputers and the scaler fit on the training fold only.
- **Not calibrated, and the notebook says so.** `class_weight="balanced"` inflates absolute probabilities, so the output is a priority ranking, not a dollar forecast.

---

## Two honest results I kept in

**Model 1's low score is the correct answer, not a failure to tune.** Within a backlog month, whether a notice slips past day 21 depends on where it landed in a queue the team was already behind on. Nothing printed on the notice determines it. So the model can identify which *periods* are dangerous but not which *notice* within one — and no tuning changes that, because the information does not exist in the data. It is also blind to the mid-2025 backlog, which was caused by a staff departure with no equivalent in the training period. That makes Model 1 a **diagnostic**: it demonstrates that deadline risk is structural, which is why the answer is a capacity fix rather than a smarter score.

**The model barely beats a deliberately stupid rule.** Before claiming triage value, I benchmarked against sorting the queue by penalty amount, biggest first — no data science, one minute in Excel:

| Strategy | top 10% | top 20% | top 30% |
|---|---|---|---|
| Expected cost (both models) | **50.5%** | 66.4% | 78.8% |
| Penalty amount only (no model) | 43.1% | **67.7%** | **86.2%** |
| Arrival order (no triage) | 10.0% | 20.0% | 30.0% |

At the 20% and 30% marks the trivial rule wins. Both crush arrival order, so triage itself is a large genuine win — but the modelling does not earn its keep on penalty risk, because 90% of queue cost is absorbed penalties driven by capacity, where amount-sorting already captures the available value.

Where it does earn its keep: the 680 uncollected billables at a $19 median value. Any amount-based rule buries those at the bottom of the queue forever. Model 2 separates them at AUC 0.681, and it is the only practical way to prioritise thousands of small items that are individually not worth a human decision.

**The honest summary: Model 2 is worth deploying on the small-ticket queue. Model 1 is a diagnostic. The headline cost problem is a staffing problem that no model was ever going to solve.**

---

## Data pipeline

1. **Generate** — 132 vehicles, 3,200 hirers, ~8,200 non-overlapping rental agreements, ~7,900 notices across 25 months, with realistic mess deliberately added.
2. **Clean** — plate normalisation (skipping it silently loses ~11% of notices), then 203 records quarantined *with a reason attached* rather than deleted: 117 duplicates, 63 missing dates, 14 impossible dates, 9 withdrawn zero-value notices.
3. **Match** — notice to hire by registration plus offence timestamp inside the hire window, asserted to resolve at most one hire per notice.
4. **Analyse** — root-cause attribution, time series, segment analysis, opportunity sizing.
5. **Model** — two models, combined into one expected-cost work queue with routing.

---

## Limitations

- **Synthetic data has synthetic structure.** I built the causal relationships into the generator, so the analysis partly rediscovers what I planted. The honest claim is that this demonstrates method and domain reasoning, not discovery. Real data would be noisier and both models would score lower.
- **Model 1 cannot predict a staffing decision**, and the notebook shows this explicitly rather than hiding it.
- **Neither model is calibrated.** They rank; they do not forecast dollars.
- **The model-based queue does not clearly beat a trivial baseline** on the bulk of the cost. That comparison is in the notebook rather than only the favourable framing.
- **No dispute or appeal process is modelled.** In reality some penalties are withdrawn on appeal.
- **Opportunity sizing is an estimate.** The 80/60/50% success rates are stated assumptions so they can be argued with.

---

## What I would do next

- Cost per vehicle per day on fleet, so branches can be compared on a fair basis.
- Survival analysis on time-to-collection rather than a binary outcome.
- A weekly capacity model: given forecast notice volume, how many processing hours keep the missed-deadline rate under 2%? That is the question the analysis actually raises, and the one Model 1 cannot answer.

---

## Running it

`fleet_infringement_analytics.ipynb` is saved with all outputs and charts already embedded, so GitHub renders the full analysis above — no need to run anything to read it.

To run it yourself: download it, open it in Google Colab, then Runtime → Run all. Takes about a minute and nothing needs installing, since Colab already has pandas, numpy, scikit-learn, matplotlib and seaborn. The generator is seeded, so every figure reproduces exactly.

The notebook creates its own data in Part 1 — there is no dataset to download. Cells must be run in order, because one seeded random generator feeds the whole notebook.
