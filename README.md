# Fleet Infringement Cost Leakage & Recovery Triage

**Where a rental fleet loses money on traffic fines and tolls, why, and which notices to work first.**

Python · pandas · matplotlib · scikit-learn

---

## About this project

I process fines, tolls and infringement notices for a fleet of 120+ Australian vehicles as part of my day job, including direct liaison with Revenue NSW.

That data is confidential and does not appear here. Instead I rebuilt the workflow from the real rules — the statutory nomination window, the liability transfer, the toll rebill mechanics — as a data generator, then analysed the result as if it were a register I had been handed.

**The domain knowledge is real. The data is synthetic.** Both of those statements matter, and the limitations section says exactly what that means for the conclusions.

---

## How liability actually works

This is the part most analyses of this problem get wrong. **Fines and tolls behave in opposite ways.**

**Infringements** (speed, red light, parking, bus lane, mobile phone) arrive in the company's name, because the company is the registered owner. There is then a 21-day window to nominate the hirer who was driving.

Nominate in time and **liability transfers completely**. The penalty moves into the hirer's name and the authority pursues them directly. It makes no difference whether that person has since flown home, ignores email, or left no card on file — that is now between them and Revenue NSW. The company's exposure is zero.

Miss the window and **the company pays the fine itself, permanently, with no appeal.**

**Tolls** work the other way. There is no nomination process. The toll operator bills the registered owner, so the company pays first and rebills the hirer afterwards. Here the money genuinely has to be collected, and a hirer who has gone overseas with no card on file is a real, permanent loss.

So the company absorbs cost in exactly three ways, and each one belongs to a different team:

| Root cause | Owner | Fixable? |
|---|---|---|
| Missed nomination deadline | Back office | Yes — it is a capacity problem |
| No hire found for the notice | Fleet ops | Mostly — movement records |
| Billable never collected | Front counter | Partly — data capture at pickup |

---

## Headline results

Over 25 months, 7,798 clean notices:

| | |
|---|---|
| Face value handled | $1,141,379 |
| **Penalties transferred to hirers** | **$922,914** (cost to company: $0) |
| Penalties absorbed | $170,508 |
| Billable raised on hirers | $161,647 |
| Billable never collected | $25,284 |
| **Total cost to business** | **$195,792** — about $7,832/month |

The process mostly works. Most penalty value is successfully transferred at no cost. The problem sits in the tail.

### Cost by root cause

| Cause | Cost | Share | Notices | Avg per notice |
|---|---|---|---|---|
| Missed nomination deadline | $112,017 | 57% | 346 | **$324** |
| No hire found | $58,491 | 30% | 416 | $141 |
| Billable not collected | $25,284 | 13% | 1,062 | **$24** |

A missed deadline costs roughly **13x** an uncollected rebill. That asymmetry drives everything downstream.

---

## Findings

**1. Cost is concentrated.** About 23% of notices generate any cost, but just 762 of them — under 10% of all notices — account for 87% of the money. That is the argument for triage over first-in-first-out processing.

**2. Liability transfer is binary.** Penalty value nominated on time: $922,914, cost to the company **$0**. Nominated late: $112,017, cost to the company **$112,017**. There is no middle ground, which is what makes the deadline worth protecting above everything else.

**3. Cost tracks the back-office backlog, not notice volume.**

| Correlation with monthly cost | Value |
|---|---|
| Missed-deadline rate | **+0.93** |
| Notice volume | +0.15 |

The worst months had unremarkable volume and average processing times pressed against the deadline. This is a staffing problem wearing a compliance problem's clothes.

**4. Volume is not cost.** Tolls are 58.5% of notices and 6.5% of the cost. Camera-detected fines are the reverse — speed cameras alone are 13.8% of volume and 32.5% of cost.

**5. A missing card on file is the strongest predictor of an uncollected bill** — and it is set at the counter, months before any notice arrives.

| Collection failure rate | Domestic | Overseas |
|---|---|---|
| Card on file | 7.6% | 23.9% |
| No card on file | 27.2% | 42.8% |

**6. 95% of the recoverable opportunity comes from process fixes, not from chasing customers harder.** Clearing the nomination backlog $89,614; daily fleet-movement reconciliation $35,095; mandating a card at pickup $7,072. Total $131,780 over 25 months, ~$63,254 annualised. The success rates behind those are stated assumptions, written as plain numbers in the code so they can be argued with.

---

## The model

**One model, scoped deliberately.** On the day a notice arrives, will the bill we raise ever be collected?

Deadline risk is **not** modelled, and that is a decision rather than an omission. Finding 3 shows missed deadlines depend on team capacity that week, not on anything printed on the notice. No model predicts a staffing decision, so the answer there is the capacity fix.

**Logistic regression**, because the question is binary and the output needs to be a probability between 0 and 1 for a priority score. `class_weight="balanced"` is essential — only ~15% of bills fail, so without it the model answers "will be collected" for everything and looks 85% accurate while being useless. Linear regression would be the wrong tool: it predicts quantities and can output negative numbers for a yes/no question.

**Validation: a time-based split.** Train on 2024 (3,445 notices), test on 2025 (3,591). A random split would let the model learn from March 2025 and then be tested on February 2025 — learning from the future, which no real deployment can do, and which flatters the score.

**Target leakage is guarded.** Only columns known on day one go in. Anything known after the outcome — `status`, `billed_recovered`, `processing_days`, `company_cost` — is excluded.

**Result: AUC 0.731** on held-out 2025 data.

### What it learned

The coefficients are readable, which is the reason to prefer a simple model — you can hand this to a manager and they can argue with it.

| Pushes risk up | | Pushes risk down | |
|---|---|---|---|
| Overseas hirer | +0.57 | Card on file | **−1.29** |
| Mobile phone offence | +0.48 | Email on file | −0.67 |
| Individual (not corporate) | +0.30 | Domestic hirer | −0.57 |

### Does it beat a stupid rule?

Before claiming a model has value, check it against something trivial — sorting the queue by the amount at stake. No data science, one minute in Excel.

| Strategy | top 10% | top 20% | top 30% |
|---|---|---|---|
| **Model risk score** | **26.7%** | **48.1%** | **62.4%** |
| Amount only (no model) | 17.7% | 30.3% | 46.3% |
| Arrival order | 10.0% | 20.0% | 30.0% |

The model wins clearly on this queue. But note what the queue contains: only notices where a bill was raised, which is about 13% of total cost. The absorbed penalties are not in it, because those are fixed by having enough staff to hit the deadline rather than by chasing anyone.

**So the honest summary: the model is useful, and it is not the answer to the headline number.**

---

## How the notebook is organised

| Part | Stage | What happens |
|---|---|---|
| 1 | **Data generation** | Build the dataset from the real business rules |
| 2 | **Exploratory data analysis (EDA)** | Look at the raw register, profile the distributions, and find what is wrong with it |
| 3 | **Data cleaning and preparation** | Fix what EDA found, match notices to hires, build the cost columns |
| 4 | **Descriptive analytics** | What happened, how much it cost, and who owns each cause |
| 5 | **Predictive analytics** | Score which bills will never be collected, and check the score is worth having |

EDA comes **before** cleaning on purpose — the exploration is what reveals which problems exist (874 notices lost to plate formatting, 116 duplicates, 14 impossible dates, 9 withdrawn), and the cleaning step then fixes exactly those.

---

## Limitations

- **Synthetic data has synthetic structure.** The causal relationships were built into the generator, so the analysis partly rediscovers what I planted. The honest claim is that this demonstrates method and domain reasoning, not discovery. Real data would be noisier and the model would score lower.
- **Deadline risk is not modelled at all,** because it depends on team capacity rather than anything on the notice.
- **The model is not calibrated.** It ranks; it does not forecast dollars.
- **The model only addresses about 13% of the cost.** It beats a trivial baseline on its own queue, but that queue is the small-ticket collection problem, not the headline number.
- **No dispute or appeal process is modelled.** In reality some penalties are withdrawn on appeal.
- **The opportunity sizing is an estimate.** The 80/60/50% success rates are stated assumptions.

---

## What I would do next

- Cost per vehicle per day on fleet, so branches can be compared on a fair basis.
- A weekly capacity model: given forecast notice volume, how many processing hours keep the missed-deadline rate under 2%? That is the question the analysis actually raises.
- Track how the recovery rate changes after a card-on-file policy is introduced.

---

## Running it

`fleet_infringement_analytics.ipynb` is saved with all outputs and charts embedded, so GitHub renders the full analysis above — nothing needs running to read it.

To run it yourself: download it, open it in Google Colab, then Runtime → Run all. Takes about a minute and nothing needs installing. The random seed is fixed, so every figure reproduces exactly.

The notebook creates its own data in Part 1 — there is no dataset to download. Cells must be run in order.
