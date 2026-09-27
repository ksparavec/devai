# Statistics primer -- how to read the experiment results in these docs

This appendix explains the statistical ideas behind every experimental
number in this repository's documentation: benchmark scores, KV-cache
comparisons, throughput and latency figures, and the laya trainer's
quality checks. It is written for a reader with some background in
statistics (an introductory course is enough) who wants to know what a
number here does and does not support.

The result pages -- [`bench-results.md`](bench-results.md),
[`backends.md`](backends.md), [`multi-token-prediction.md`](multi-token-prediction.md),
[`laya-trainer.md`](laya-trainer.md) and the others -- use the vocabulary
and the conventions defined here. Where they report an interval, a test
or a percentile, this page says which one and why.

**Two meanings of "sampling".** In this repository the word is used in
two unrelated senses:

- *Decoding sampling*: how a language model picks its next token from a
  probability distribution (temperature, top-p, ...). That is the subject of
  [`sampling-strategies.md`](sampling-strategies.md).
- *Statistical sampling*: how a subset of items (questions, prompts,
  documents) is drawn from a larger population for measurement. That is
  the subject of this page.

They meet in one place, explained in Sec. 2: when decoding sampling is
stochastic, every scored answer is itself a random draw.

All numerical examples below were computed with a stdlib-only script,
`scripts/stats/primer_examples.py`, and can be reproduced exactly.

---

## Main results at a glance (as of 2026-09-27)

Each line states a headline result in the strict form its page supports,
with the page that gives the details.

- **Benchmark leaderboard** ([bench-results.md](bench-results.md)).
  - **What was run:** nine vLLM models in one sweep on 2026-05-05, one run
    each, at each engine's default (stochastic) sampling.
  - **Result:** no difference among the top four (Nemotron-3-Nano,
    Qwen3-14B, Qwen3-8B, gpt-oss-20b) is established after correcting for
    the 36 pairwise comparisons (smallest raw p = 0.024; Holm-adjusted
    0.26). Two rows score below all others.
  - **Limits:** time-outs were scored as wrong answers, and the HumanEval
    scorer has a known defect.
- **q8_0 vs f16 KV cache** ([backends.md](backends.md), "Per-tier
  KV-cache dtype").
  - **Result:** on 60 GPQA items, 0.767 and 0.750 (two q8_0 runs) against
    0.867 (f16). The paired differences are -0.100 (95% CI -0.222 to
    +0.022) and -0.117 (-0.255 to +0.026), p = 0.18 and 0.17.
  - **Limits:** the net gap is more than accounted for by extra time-outs
    under q8_0, and the tiers also differed in context length. A quality
    cost is neither shown nor excluded.
- **vLLM vs SGLang quality** ([backends.md](backends.md)): no difference
  established in 18 paired comparisons, which are confounded by context
  length, KV dtype and date.
- **MTP speculative decoding**
  ([multi-token-prediction.md](multi-token-prediction.md)).
  - **Result:** decode was 2.52x (95% CI 2.42-2.61) and 2.60x (2.50-2.70)
    faster on the two Qwen3.8-27B builds, paired over 34 prompts.
  - **Limits:** one run per arm; the interval covers prompt-to-prompt
    variation only.
- **Decode speed vs memory bandwidth**
  ([llm-tokens-and-speed.md](llm-tokens-and-speed.md)).
  - **Utilisation:** single-stream decode reached 79-89% of a ceiling
    derived from an unsourced 640 GB/s peak.
  - **Paired test of the model:** two builds differing 8% in bytes per
    token decoded 6.9% apart. The predicted ratio is 1.080; the observed
    ratio is 1.069 (95% CI 1.067-1.071, prompt-to-prompt only, one run per
    build). Bytes explain most of the difference, but not all.
- **laya student for sentiment polarity**
  ([laya-trainer.md](laya-trainer.md)).
  - **Gate:** the installed round passed a 0.90 precision target. It was
    chosen as the best of three rounds on one held-out set; the bound
    adjusted for that choice is 0.893, below 0.90.
  - **Fresh documents:** among the documents the student answered, it
    agreed with the teacher on 117/123 = 0.951 (one-sided 95% lower bound
    0.906), at a coverage of 0.246.
  - **Opinion text alone:** 69/75, with a lower bound of 0.848, so 0.90 is
    not shown there.
  - **Estimand:** all of these are agreement with the teacher, not
    correctness.
- **Cold start** ([nvfp4-coldstart.md](nvfp4-coldstart.md)): the 27B
  teacher went from launch to ready in a median of 125 s (13 launches) with
  its FlashInfer cache warm, and 261-278 s with empty caches.

---

## 1. What an experiment here estimates

Every measurement in these docs answers one of three questions:

1. **Estimation**: how large is a quantity? ("What fraction of GPQA
   questions does model M answer correctly?")
2. **Comparison**: do two configurations differ? ("Does q8_0 KV cache
   lower GPQA accuracy relative to f16?")
3. **Decision**: which option should be used, given a rule? ("Which
   confidence threshold does the student model need before its answers
   are accepted?")

To interpret any of them you need four things, and a result page states
all four next to the number:

- **Population** (the *inference target*): the set of things the claim
  is about. For a benchmark it is usually the full benchmark (for
  example the 198 questions of GPQA Diamond). It can also be a fixed,
  named item set ("the first 100 GSM8K test questions"), in which case the
  claim is about those items only.
- **Sample and sampling frame**: which units were actually measured and
  how they were chosen. The two methods used here behave very
  differently:
  - a *seeded random sample* (the MMLU-Pro and GPQA subsets: shuffle
    with a fixed seed, take the first n). A random sample supports
    conclusions about the whole benchmark;
  - the *first n items* in the file's order (the GSM8K, HumanEval and
    HumanEval+ subsets). A first-n subset supports conclusions about
    those n items.
    Extending them to the full benchmark requires the untestable
    assumption that the first n are representative;
  - the *whole set* (the 20 hand-written tool-use prompts): the result
    describes those prompts, and the only randomness is decoding.
- **Sources of randomness**: what would change if the experiment were
  repeated. Here these are item selection, stochastic decoding (Sec. 2),
  run-to-run system noise (for timings), and the training seed (for
  laya students).
- **What was held fixed**: one host, one GPU, one engine version, one
  checkpoint, and often **one run**. A single run measures one
  realisation. It says nothing directly about the spread between runs,
  and every page flags this where it applies.

---

## 2. Scores and the binomial model

Most quality numbers are **proportions**: the number of successes x out
of n items.

| Score | Unit | Success means |
| --- | --- | --- |
| GSM8K, MMLU-Pro, GPQA accuracy | question | the final answer matches the key |
| HumanEval / HumanEval+ `pass@1` | programming problem | the one generated program passes every unit test |
| tool-use score | prompt | exactly one call to the expected tool, with exactly the expected arguments (all or nothing) |
| laya precision, agreement rates | segment or item | the student's answer equals the reference |

(The leak rate looks like a proportion but is not one; see the next paragraph.)

**Count rates are not proportions.** The leak probe counts reasoning-marker
*matches* across all responses and divides by the number of prompts. One
prompt can contain several matches, so "0.075" means 3 matches in 40 prompts,
not that 7.5% of prompts leaked. A count is modelled as Poisson. Its exact
confidence interval (Sec. 3 explains what that is) is **Garwood's** (1936): for 3 matches, [0.619, 8.767] matches, or
[0.015, 0.219] per prompt over 40 prompts. How many *prompts* were affected
(1 to 3 here) is a different quantity, and it needs per-prompt records.

**Composite scores.** An "aggregate" that averages proportions measured on
different n (100, 50 and 20 items) has no standard interpretation. Its
variance is dominated by the smallest component. Where one is shown, its
interval comes from a **stratified bootstrap** that resamples items within
each task, and the components are reported alongside it.

`pass@1` here means *one* sample per problem, scored pass/fail. It is
the k=1 case of the pass@k estimator of Chen et al. (2021) and is simply
a proportion. So is the tool-use score: each prompt scores 1 or 0, with
no partial credit, and the page reports x/20. Rates such as tokens per
second are **not** proportions and need other methods (Sec. 9).

**The model.** The standard assumption is that each item i has a
probability p_i of being answered correctly and that outcomes are
independent across items. Two cases matter:

- **Deterministic decoding** (greedy, temperature 0): each p_i is 0 or
  1. The only randomness is which items were sampled.
- **Stochastic decoding** (temperature > 0): p_i can lie strictly
  between 0 and 1. The same model can answer the same question right on
  one run and wrong on the next.

The benchmark harness in this repository intended temperature 0, but
the setting never reached the model (see `bench-results.md`,
"Methodology"). Every stored benchmark score was therefore measured
under each backend's default, stochastic decoding. The consequences are
spelled out below where they matter.

The quantity estimated is the mean success probability over the target
items, p = mean(p_i). Its estimate is the observed proportion
p_hat = x / n.

---

## 3. Confidence intervals for a proportion

A **95% confidence interval** is the output of a procedure that, over
repeated experiments, covers the true value in (at least, or about) 95%
of cases. It is *not* a 95% probability that the true value lies in this
particular interval; that statement requires a Bayesian model with a prior.
Read it as "the values of p that are compatible with what we observed,
at the 5% level" (tests and p-values are defined in Sec. 5).

**Clopper-Pearson (exact).** This is the primary interval in these docs.
It inverts two one-sided exact binomial tests: the lower bound is the p
at which observing x or more successes has probability 2.5%, and the
upper bound is the p at which observing x or fewer has probability 2.5%.
Its coverage is guaranteed to be at least 95% for every p, so it is
*conservative*, somewhat wider than necessary. It behaves correctly at
x = 0 and x = n, where simpler intervals fail.

**Wilson score.** This interval inverts the score test. Its coverage is
close to 95% on average but can dip below it for particular p. Some
machine-readable outputs report it alongside Clopper-Pearson.

**One-sided bounds.** Some claims point in one direction only, such as
"precision is at least 0.90". For those, a **one-sided 95% lower
bound** is the natural statistic: the smallest p still compatible with
the data at the 5% level, with no upper limit. A one-sided 95%
Clopper-Pearson lower bound equals the lower end of the *two-sided 90%*
interval, so it is always higher than the lower end of the two-sided
95% interval. For example, 81 of 84 gives a one-sided 95% lower bound
of 0.910, but a two-sided 95% interval of [0.899, 0.993]. Each page
says which it reports. When every observation is a success, the
one-sided lower bound is alpha^(1/n): at least 59 successes out of 59
are needed before the bound clears 0.95.

**Why not "p_hat +- 1.96 x standard error" (the Wald interval).** It
undercovers badly at small n and collapses to a zero-width interval at
x = 0 or x = n. It is not used anywhere in these docs.

Worked example: 47 correct out of 60 GPQA questions.

```
p_hat          = 47/60 = 0.783
Clopper-Pearson  [0.658, 0.879]
Wilson           [0.664, 0.869]
```

The interval is about 22 points wide. **Sample size controls the
width.** Clopper-Pearson intervals at p_hat = 0.80:

| n | x | 95% Clopper-Pearson | width |
| --- | --- | --- | --- |
| 20 | 16 | [0.563, 0.943] | 0.38 |
| 50 | 40 | [0.663, 0.900] | 0.24 |
| 60 | 48 | [0.677, 0.892] | 0.22 |
| 100 | 80 | [0.708, 0.873] | 0.17 |
| 198 | 158 | [0.735, 0.852] | 0.12 |
| 1000 | 800 | [0.774, 0.824] | 0.05 |

The subsets used by the benchmark harness have n between 20 and 100. At
these sizes a single score is precise to roughly +-8 to +-19 points. Two
models whose scores differ by less than that are usually not
distinguishable from their scores alone (Sec. 4).

**Edge cases.** 60 out of 60 gives [0.940, 1.000]. The data are
compatible with a true rate as low as 94%, so 100% on a subset is not
proof of perfection. A leak rate of 0 out of 40 gives [0.000, 0.088]:
"no leak observed in 40 prompts" is compatible with a true leak rate of
up to about 9%.

**Finite-population correction (FPC).** GPQA Diamond has N = 198
questions, and the harness samples n = 60 of them without replacement.
When the target is those 198 questions *and decoding is deterministic*,
the variance of p_hat shrinks by the factor (N - n) / (N - 1), and the
interval half-width by its square root:

```
sqrt((198 - 60) / (198 - 1)) = 0.837
```

That is 16% narrower. Under **stochastic** decoding the correction does
not apply in full. The variance of p_hat has two parts:

```
Var(p_hat) = (N - n)/(N - 1) x S2_items / n   +   mean_i[ p_i (1 - p_i) ] / n
             (which items were drawn)             (how each draw was decoded)
```

The FPC shrinks only the first part. S2_items is the variance of the
p_i across the N items of the population (divisor N). Because the benchmark scores here were
measured under stochastic decoding, the uncorrected Clopper-Pearson
interval is the one reported. It is valid and conservative. An
FPC-corrected interval, where shown, is labelled "valid only under
deterministic decoding".

**First-n subsets.** For a first-n subset the target is the mean p_i of
those n items under the decoding actually used. A sum of independent
Bernoulli variables with unequal p_i is less spread out than a binomial
variable with the same mean: its variance is never larger, and Hoeffding
(1956) showed its tail probabilities are bounded by the binomial ones
over the range an interval uses. So Clopper-Pearson stays conservative
for that target, except possibly when p is very close to 0 or 1.

**Time-outs: bounds instead of a point.** When t of n items were cut off
by a time limit and scored wrong, the data only bound the score.

- **The observed x/n is a lower bound.** It is exact if every time-out
  would have been wrong.
- **(x + t)/n is an upper bound.** It is exact if every time-out would have
  been right.

For example, 74 correct and 24 time-outs out of 100 bound the rate between
0.74 and 0.98. The accuracy among completed items (74/76 = 0.974, 95%
interval [0.908, 0.997]) is **not** an estimate of the full rate, because
hard items are the ones most likely to time out. Reporting the bounds
together with the time-out count is the honest summary. This approach is
called *partial identification*.

The bounds describe this one run. To add sampling uncertainty, take the
Clopper-Pearson lower bound of x/n and the Clopper-Pearson upper bound of
(x + t)/n. Together they cover the whole range with at least 95%
probability, because each end fails with probability at most 2.5%. For the
example that gives [0.643, 0.998]. Imbens and Manski (2004) give a tighter
interval.

---

## 4. Comparing two configurations

"Model A scored 0.80, model B scored 0.70" is not yet a comparison. The
question is whether the difference is larger than the noise, and that
depends on the **design**.

### 4.1 Paired designs and McNemar's test

When both configurations answered **the same items** (the same seeded
subset), each item gives a pair of outcomes. Only the items on which the
two disagree carry information about the difference:

```
                  B correct   B wrong
    A correct        a           b
    A wrong          c           d
```

The **exact McNemar test** asks whether b and c are consistent with a
fair coin. Under the null hypothesis of no systematic difference, each
discordant item is equally likely to fall either way, so given
m = b + c, b ~ Binomial(m, 1/2). The two-sided p-value is
2 x P(X <= min(b, c)), capped at 1.

Worked example. A and B are run on the same 60 questions. A is right and
B wrong on b = 8 of them, and the opposite on c = 2:

```
m = 10,  p = 2 x P(X <= 2 | Binomial(10, 0.5)) = 0.109
```

The accuracies differ by (8 - 2)/60 = 10 points, yet the test does not
reject at the 5% level. A split of b = 7 against c = 0 gives p = 0.016,
because a smaller total but a lopsided split is stronger evidence.

The accompanying interval for the paired difference p_A - p_B is
**Newcombe's hybrid score interval for paired data** (Newcombe 1998b,
method 10). It is built from Wilson intervals and accounts for the
correlation between the paired outcomes.

**How much pairing helps depends on that correlation.** When the same
items are easy or hard for both configurations, pairing cancels the
item-to-item variation and narrows the interval a lot. Stochastic
decoding weakens that correlation, because an item can go either way on
any run. In the KV comparison here the paired interval was [-0.222,
+0.022] against [-0.237, +0.040] unpaired, which is only slightly
narrower.

**Stochastic decoding and one run per configuration.** An item answered
right under A and wrong under B may reflect decoding noise rather than a
real difference between A and B. The McNemar test stays valid for the
null hypothesis of *no systematic difference*. That null implies the
same p_i under A and B for every item, which makes the two discordant
cells equally likely. But one run per configuration cannot separate
"A is better on this item" from "this item is a coin flip for both". Only
repeated runs can.

### 4.2 Unpaired designs

When the two configurations were measured on **different** items, or
the item identities cannot be matched, the comparison is between two
independent proportions. The interval for the difference is
**Newcombe's hybrid score interval** for independent samples (Newcombe
1998a, method 10). Unpaired comparisons need more items than paired ones
to detect the same difference. How many more depends on the within-item
correlation, which is modest under stochastic decoding (Sec. 4.1).

### 4.3 Confounding

A comparison isolates a factor only if everything else was held equal.
If the q8_0 and f16 cells also differ in context length, flash-attention
setting, date, engine version or item set, a difference between them
cannot be attributed to the KV dtype alone. The result pages list every
known difference between compared cells. Where more than one factor
changed, the comparison is labelled **confounded**.

### 4.4 The test-retest noise floor

Running the *same* configuration twice on the same items shows how much
of a difference is noise. Under stochastic decoding, two identical runs
disagree on some items. Two runs of one q8_0 configuration here disagreed
on 13 of 60 GPQA items. That count measures the noise *level*, and it was
similar to how often either run disagreed with the f16 run (14 and 19
items).

The quantity to compare, though, is the **net** difference: the retest pair
differed by 1 item (6 against 7 discordant), while q8_0 and f16 differed
by 6 and 7 items. Whether a net difference is larger than chance is what
McNemar's test assesses, using the discordant items. A single retest pair
also gives only a rough idea of the noise level, not a precise floor.

### 4.5 Randomization tests and the bootstrap

Two further paired methods appear on the result pages.

**The paired sign-flip (randomization) test.** Use it when the quantity
compared is not a single proportion, for example an aggregate averaging
several tasks. For each item, take the difference between configurations
A and B. If A and B do not differ systematically, the sign of each
difference is as likely to be + as -. The test flips the signs at random
many times (10 000 here, with a fixed seed), recomputes the statistic
each time, and reports the fraction of flips at least as extreme as the
observed value. That fraction is the p-value. It needs no distributional
assumption beyond the pairing.

**The paired bootstrap.** For an interval rather than a test, resample the
*items* with replacement (keeping each item's A and B outcomes together),
recompute the difference, repeat 10 000 times, and take the 2.5th and
97.5th percentiles. For a composite that averages tasks, the resampling
is done within each task (a *stratified* bootstrap, Sec. 2).

---

## 5. p-values, power and what "not significant" means

**p-value.** The probability, computed under the null hypothesis, of a
result at least as extreme as the one observed. A p-value below 0.05 is
reported as "significant at the 5% level". It is not the probability
that the null hypothesis is true, and it does not measure how large or
important the effect is.

**Statistical versus practical significance.** A difference can be
statistically detectable but too small to matter, or large enough to
matter but not detectable at the sample size used. The docs report the
interval for the difference so that you can judge both.

**"Not significant" is not "no difference".** Failing to reject means the
data are compatible with no difference. They are usually also compatible
with a substantial difference. The upper end of the confidence interval
for the difference says how large a difference the data still allow.
Where a page says "not distinguishable at the 5% level (p = ...)", read
it that way, and never as "equal".

**Power and the minimum detectable effect.** Power is the probability
that a test detects a difference of a given true size. For two
independent proportions of 0.80 and 0.70, measured on n items each and
tested at the 5% level:

```
n = 60 per arm      power ~ 0.24
n = 300 per arm     power ~ 0.81
n = 294 per arm     is needed for 80% power
```

(normal approximation). A 10-point difference between two models
benchmarked on 60 unpaired questions is missed about three times in
four. Paired designs need fewer items when outcomes on the same item are
strongly correlated. Under stochastic decoding that correlation is weaker,
and the gain is smaller (Sec. 4.1).

**Equivalence and non-inferiority.** A claim such as "fp8 costs at most
2 points" is a statement about an upper bound, not a failure to find a
difference. It needs a pre-stated *margin* (here 2 points) and is
supported only if the upper end of the confidence interval for the loss
lies below that margin. The usual level is a one-sided 95% bound, which is
the same as a two-sided 90% interval (the "two one-sided tests", or TOST,
procedure). On a 60-question subset the interval for a
difference is typically +-10 points or wider, so a 2-point margin cannot
be established at that size.

---

## 6. Multiple comparisons

Testing many hypotheses that are in fact true (no real difference) at
the 5% level produces false positives at about 5% per test. When a claim
says that *some* difference exists -- "model A beats B", or a ranking of
eight models built from pairwise wins -- the chance of at least one false
positive grows with the family's size. That is what a multiplicity
correction controls.

Two kinds of claim need **no** such correction:

- **"All of them pass".** For example, "fp8 is non-inferior on every one of
  seven metrics" holds only if each of the seven tests passes on its own.
  This is an *intersection-union* test: requiring every test to pass is
  already strict, so no adjustment is needed.
- **"None of them differs".** This is a statement about the *absence* of
  evidence. Correcting for multiplicity only makes non-rejection easier,
  so it cannot support such a claim. The pages give the raw p-values,
  together with the intervals or the power, for claims of this kind.

The docs control the family-wise error rate with **Holm's step-down
procedure** (Holm 1979):

1. Sort the m p-values: p(1) <= p(2) <= ... <= p(m).
2. Compare p(k) with 0.05 / (m - k + 1), starting at k = 1, and stop at
   the first one that is not smaller. That hypothesis and all later ones
   are not rejected.
3. Equivalently, the adjusted p-values are
   p_adj(k) = max over j <= k of min(1, (m - j + 1) x p(j)).

Worked example with m = 4:

| raw p | Holm-adjusted p | rejected at 5%? |
| --- | --- | --- |
| 0.004 | 0.016 | yes |
| 0.020 | 0.060 | no |
| 0.030 | 0.060 | no |
| 0.40 | 0.40 | no |

The second comparison, at p = 0.020, would pass on its own but not as
one of four. When the result pages claim that differences exist, they
name each family and give both p-values.

---

## 7. Designs that void the nominal error rate

A confidence interval or p-value promises a 5% error rate only if the
analysis was fixed before the data were seen, and only if the data
arose as the model assumes. The following patterns break that promise.
Each has occurred in this repository, and the result pages point them
out where they apply.

- **Choosing among candidates after looking.** When the best of k
  candidates (training rounds, thresholds, models) is picked on the same
  data that then certifies it, the winner's confidence bound is too
  optimistic. The error rate applies to one pre-chosen candidate, not to
  the best of k. The standard repair is a *simultaneous* bound, for
  example Bonferroni: compute each bound at alpha / k. Fresh data avoid
  the problem altogether, and the pages report such confirmations
  separately.

  Changing the *target* after seeing the data is a different matter. A
  valid lower confidence bound does not depend on the target it is
  compared with, so the chance of a false pass stays at most 5% whatever
  target is chosen. Lowering the target changes the *requirement* (what
  counts as good enough); that is a decision for the owner, not an
  error-rate problem.
- **Selected samples.** A sample chosen by the model's own confidence
  (for example, the rows left over after the least-confident ones were
  removed) favours the model by construction. It cannot estimate
  performance on typical data, whatever its size.
- **Informative truncation.** If an item that runs out of time is scored
  as *wrong*, the score mixes two things: how often the model is right,
  and how often it is too slow under the harness's time limit and
  queueing. The two must be reported separately. When the arms of a
  comparison had different numbers of time-outs, the comparison cannot
  separate correctness from speed. That holds even when the time-outs may
  themselves be an effect of the treatment (a configuration that produces
  runaway generations). Some
  benchmark rows here contain dozens of time-outs; see
  `bench-results.md`.
- **Dependent items.** The binomial model assumes independent items.
  Requests that share one serving slot are not independent: one very
  long generation delays the ones queued behind it, and several can
  time out together. A test computed on such outcomes assumes an
  independence that does not hold. Positive dependence usually makes the
  true uncertainty larger than the computed one, so "not distinguishable"
  survives, but a small p-value computed this way should not be trusted.
- **Conditioning on an outcome.** Restricting a comparison to "items
  that completed in both runs" conditions on something the treatment
  may have caused. Such an analysis is descriptive, not a test of the
  treatment.
- **Choosing which run to keep.** "The first run looked off, so it was
  re-run, and the re-run is reported" is a selection procedure. Its
  error rate is unknown. Report every run, or pre-specify that runs are
  averaged.
- **Explaining after the fact.** A mechanism proposed to explain a
  difference that turned out not to be significant ("the long chains
  accumulate quantization error") is a hypothesis for a new experiment,
  not a finding.

---

## 8. Percentiles and quantiles

The **q-quantile** of a distribution is the value below which a fraction q
of it lies. The median is the 0.5-quantile. The **p50, p90, p95** of a set
of timings are its 0.50, 0.90 and 0.95 sample quantiles. **Tail latency**
means the high quantiles (p90, p95, p99): the latency that 10%, 5% or 1%
of requests exceed.

**Why medians and high quantiles, not means.** Latency distributions are
skewed to the right: a few slow requests (a cold start, a long
generation, a queued request) pull the mean far above the typical value.
The median describes a typical request, and the high quantiles describe
the bad cases a user notices.

**Sample quantiles have several definitions.** For a small sample they
give different answers. With the n = 24 sorted timings x(1) <= ... <=
x(24):

| Definition | Used by | p90 is |
| --- | --- | --- |
| Nearest rank: x(ceil(q x n)) | common textbook definition | x(22) |
| Linear interpolation (Hyndman-Fan type 7) | NumPy and R defaults; Python `statistics.quantiles(method="inclusive")`; the bench harness | x(21) + 0.7 x (x(22) - x(21)) |
| Hyndman-Fan type 6 | Python `statistics.quantiles` default (`method="exclusive"`) | x(22) + 0.5 x (x(23) - x(22)) |
| Index int(q x (n - 1)), 0-based | aiagent's t/S script (`~/laya-pilot/measure_ts.py`, not in this repo) | x(21) |

Each result page names the definition its producing code uses. With n
this small, **a high quantile rests on the top few observations**: the
p90 of 24 values is determined by the second- to fourth-largest. It is
not stable and should not be compared across runs without an interval.

**An interval for the median.** This one needs no distributional
assumption. For n observations, the interval between the order
statistics x(r) and x(n - r + 1) covers the true median with probability
1 - 2 x P(Binomial(n, 1/2) <= r - 1). For 95% or more:

| n | interval | exact coverage |
| --- | --- | --- |
| 24 | [x(7), x(18)] | 97.7% |
| 60 | [x(22), x(39)] | 97.3% |
| 100 | [x(40), x(61)] | 96.5% |

This requires the raw per-request values. Where only a summary (a
median, a p90) was stored, no interval can be computed, and the pages
say so.

**High quantiles need more data.** A *distribution-free* 95% interval for
the q-quantile exists only when the widest possible interval, from the
sample's minimum to its maximum, covers it with at least 95% probability:
1 - q^n - (1 - q)^n >= 0.95. That means n >= 29 for a p90 and n >= 59 for
a p95. With 39 steady-state prompts per run, the bench's p95 latency has
no distribution-free interval. It describes that run only.

---

## 9. Rates, throughput and ratios

**Per-request versus aggregate.** "Tokens per second" means two
different things:

- **decode rate per request**: the tokens one request generated, divided
  by its generation time. It describes the speed one user sees;
- **aggregate throughput**: the tokens all concurrent requests generated,
  divided by wall-clock time. It describes the machine's total output.

With 4 requests in flight, the aggregate can be 3-4 times the
per-request rate while each request individually slows down. The two
are not interchangeable, and the pages always say which is meant.

**What counts as a token** matters as much as the timing. Engine-reported
token counts are exact. A characters/4 estimate is an approximation whose
error depends on the tokenizer and the text.

- **What was compared here.** The bench's median rate (characters/4,
  client timing, all 40 prompts) against the engine-counted median rate.
  The two agreed within about 2% when the engine median was taken over
  outputs of at least 128 tokens, and differed by 0-7% (ratios 0.93-1.01)
  against the engine median over all prompts. That compares two summaries.
- **What was not compared.** Per-request token counts were never compared,
  because the harness does not store its per-request character counts.
- **Code.** One unretained measurement on code read 10-25% low. Every bench TPS value here is such an
estimate (see `bench-results.md`). Reasoning tokens may or may not be
included, and each page says which.

**Ratios and speedups.** A speedup S = T_sequential / T_parallel is a
ratio of two random quantities. Its uncertainty comes from both.
Without replication neither is known. With raw per-call data, a
**percentile bootstrap** gives an interval: resample the calls with
replacement B = 10000 times with a fixed seed, recompute S each time,
and take the 2.5th and 97.5th percentiles. Two design points strengthen
a speedup measurement:

- the same inputs in both arms (paired). Otherwise differences in input
  length are confounded with the speedup;
- a known ceiling. With at most k requests served at once, S <= k.

**Drift, warm-up and replication.** System timings change over a
session: caches warm up, clocks and temperatures vary, and other
processes compete. A fair performance comparison excludes warm-up
requests and interleaves the two conditions (A B A B ...) rather than
running them back to back. It also repeats the whole measurement in
more than one session. A single run of each condition, run
consecutively, is labelled as such.

**The unit of replication.** Forty requests in one run are forty
observations of *that run*. They show how prompts differ, not how runs
differ. A run can shift as a whole: different cache state, clocks, a
different engine launch. When single runs of the same configuration were
repeated here, eight models re-run within a few days showed a typical
run-to-run coefficient of variation of about 1.3% (95% CI roughly
0.8-2.6%). Across months, images or context lengths, the spread between a
model's runs reached 3-13% (max/min); the larger values coincide with
changed context lengths. An interval
computed from one run's requests covers prompt-to-prompt variation only,
and the pages say so. Treating requests as if they were independent runs
is called *pseudo-replication*.

**Summarising ratios.** For a per-prompt speedup (MTP on / off on the same
prompt), the **geometric mean** of the ratios is the natural average,
because ratios multiply. The pages give it with a bootstrap interval over
prompts. A **ratio of sums** (total tokens / total time) weights long
requests more heavily and answers a different question: throughput for
the whole workload. The two are not interchangeable.

**Decomposing a concurrency speedup.** When k requests run at once, the
speedup of a batch over sequential calls splits into two factors:

```
S = (achieved concurrency) / (per-call slowdown)
```

For example, 3.74 requests in flight on average, with each call 1.19x
slower than alone, gives about 3.14. S is capped by the engine's limit on
concurrent sequences (4 on the teacher), not by the number of requests
sent.

**Bias versus noise.** Some errors do not average out.

- **Counting errors:** estimating tokens as characters/4; counting N tokens
  over N-1 inter-token gaps, which adds about 7% at 16 tokens.
- **Sampling limits:** sampling VRAM once per second, which can only miss
  a peak.
- **Clock differences:** measuring at the client rather than inside the
  engine.

Such errors are **bias**, and replication cannot remove it. A measured
value *at or above* a physical bound, such as a decode rate at 100% of the
memory-bandwidth ceiling, is more likely to indicate such a bias, or a
wrong bound, than perfect efficiency.

**Model-based ceilings.** Some pages derive a theoretical bound rather
than measure one. An example is the decode ceiling "memory bandwidth /
bytes read per token". Such a number is exactly as good as its
assumptions (for example, "the weights are read once per token and
KV-cache reads are negligible"), and the pages state those assumptions.
A "utilisation" (measured / ceiling) inherits the uncertainty of the
measured value and the approximations of the ceiling.

---

## 10. Scores, thresholds and decisions

Many models output not only an answer but a **score**: a confidence, a
probability or a margin. A **threshold** (written tau in these docs)
turns a score into a decision: accept the answer if the score is at
least tau, otherwise abstain or hand the item to another model. In the
laya/aiagent cascade, the fast student model answers an item only when
its confidence reaches tau. Everything else goes to the slow teacher
model.

Raising tau trades volume for quality:

- **Coverage** (or acceptance rate): the fraction of items the student
  answers, #{score >= tau} / #items.
- **Precision** (among accepted items): the fraction of accepted answers
  that are correct, #{accepted and correct} / #{accepted}.

Higher tau means fewer accepted items (lower coverage) and usually
higher precision. Precision is a proportion estimated on the *accepted*
items only, so its n is the number accepted, not the size of the test
set. A precision of 0.90 on 30 accepted items has the 95%
Clopper-Pearson interval [0.735, 0.979] (27 of 30). The same 0.90 on 300
accepted items has [0.860, 0.932] (270 of 300).

**Classifying against a fixed cut-off.** A rule such as "a model is
production-ready if its tool-use score is at least 0.9" classifies by a
point estimate. When the confidence interval contains the cut-off, the
classification is not determined by the data. 17/20 has the 95% interval
[0.62, 0.97], and 20/20 has [0.83, 1.00], so a re-run could move either
model across the line. The pages mark such memberships as undetermined.

**Agreement is not correctness.** When a student model is trained to
imitate a teacher, "precision" and "accuracy" are measured against the
*teacher's* label. They are agreement rates, and they are only as good
as the teacher. When the teacher's own samples disagree on an item (as
they did on 53 of 334 held-out rows in the laya campaign), the label is
itself uncertain. Only independent human labels measure correctness.

**Choosing tau without fooling yourself.** If tau is chosen by looking
at one data set and precision is then reported on the same data set, the
reported precision is optimistically biased, because the threshold was
fitted to that data's noise. The designs here keep separate splits:

- a **training** split, to fit the model;
- a **calibration** split, to fit calibration and choose tau;
- a **held-out** (test) split, used once to estimate precision and
  coverage at the chosen tau.

Splits are **group-disjoint** when related items (for example segments
of the same document) all go into the same split, so that the held-out
estimate is not inflated by near-duplicates of training items.

---

## 11. Calibration and discrimination

A score is **calibrated** if, among all items given a confidence of
about 0.8, about 80% are correct. Calibration concerns the *meaning* of
the score's numeric value. **Discrimination** concerns its *ranking*:
whether correct items tend to get higher scores than incorrect ones. A
model can rank well and be badly calibrated (always overconfident), or
be calibrated and rank poorly. A threshold rule needs good
discrimination. A threshold stated as a probability ("accept above
0.9") also needs calibration.

**Temperature scaling** (Guo et al. 2017) is the simplest calibration
method, and the laya trainer uses it. A single scalar T divides the
model's logits before the softmax, p = softmax(z / T). T is fitted on
the calibration split by minimising the cross-entropy against the labels
the model is meant to match. For the laya students those are the
teacher's vote shares (the fraction of 3 teacher samples choosing each
option), not independently verified true labels:

- T > 1 softens overconfident probabilities;
- T < 1 sharpens underconfident ones;
- T = 1 changes nothing.

Temperature scaling never changes which option has the highest score *for
a given item*, so accuracy is unchanged. It can, however, change the
*ordering of confidences across items* when there are more than two
options. For example, the logits (2, 0, 0, 0) and (3, 2.5, -10, -10) give
top-option confidences of 0.711 and 0.622 at T = 1, but 0.332 and 0.487
at T = 5. A separate T per question type can also reorder items of
different types. So the AUROC and the set of items accepted at a given
tau can both change after calibration.

The trainer clamps T to [0.5, 5.0] and keeps T = 1 for any question type
with fewer than 10 calibration items, because a single-parameter fit on
fewer items is too noisy to trust.

This "temperature" is a different parameter from the decoding
temperature of Sec. 2, although both divide logits by a constant.

**AUROC** (area under the ROC curve) measures discrimination. It is the
probability that a randomly chosen positive item gets a higher score
than a randomly chosen negative item, with ties counted as one half.

- 0.5 means no discrimination (chance level);
- 1.0 means perfect separation.

Its uncertainty depends on the numbers of positives and negatives, not
on their total. Using the Hanley-McNeil (1982) approximation, an AUROC
of 0.64 has these 95% intervals:

| positives / negatives | 95% interval |
| --- | --- |
| 20 / 20 | [0.47, 0.81], includes chance level |
| 50 / 50 | [0.53, 0.75] |
| 200 / 200 | [0.59, 0.69] |

DeLong's method (DeLong et al. 1988) and the bootstrap are the
alternatives when the raw scores are available. The pages say which is
used. When several decisions share one input (two questions asked about
the same text), they are not independent. A **cluster bootstrap**
resamples whole texts rather than single decisions and gives an honest,
usually wider, interval.

**Comparing with a baseline.** "The student gets 6 of 14 right, the
majority-class baseline 4 of 14" invites a test. A simple choice is the
exact binomial test of 6/14 against a fixed rate of 4/14 = 0.286. Its
one-sided p-value P(X >= 6 | n = 14, p = 0.286) is 0.185, and 6/14 has
the Clopper-Pearson interval [0.177, 0.711].

That test treats 4/14 as a known constant. Here the baseline was computed
on the same 14 items (the most common label *in this sample*, an
optimistic "oracle" baseline), so the right test is the paired McNemar
test of Sec. 4.1. The laya plan reports it: p = 0.73. Either way, the data
do not show that the student beats the baseline. That is not evidence of
any improvement, and it is not evidence of equality either.

---

## 12. Deterministic checks are not statistics

Some verification steps compare two computations that should agree up
to floating-point rounding. Examples:

- the ONNX export's probabilities against PyTorch's ("parity");
- the answers a runtime must reproduce ("golden" rows);
- file hashes.

These are **tolerance checks**: a metric (for example the maximum
absolute difference between two probability vectors over a stated set
of items) is compared against a fixed tolerance (for example 1e-3).
Nothing is sampled, so there is no confidence interval and no p-value.
The result is pass or fail, and the pages report the metric's
definition, the item set it ranges over, the observed maximum and the
tolerance.

---

## 13. Engineering margins

A limit such as "hold the GPU for at most 900 s" is set from a few
observed durations times a safety factor. The three observations here
(123.3, 135.2 and 140.4 s) came from three training jobs of *different*
sizes, so they are not three draws from one distribution. Even if they
were, the largest of three values is only a 75% upper prediction bound
for the next one, and no 95% upper bound can be derived from n = 3. Such
a limit is an engineering margin. The pages report the observations
themselves and the factor applied, and call the result a margin, not an
estimate.

When a duration depends mainly on the input size, the informative
summary is a **cost model**, not a statistic.

- **The laya example:** training cost about 27 ms per training row, where
  the row count varied between jobs. A straight line through two job sizes
  (hold = 34.3 + 89.0 k s, with k the dataset size relative to the first
  campaign round) is consistent with the observations.
- **How much the data support it:** little. A line through two points
  leaves nothing to test the fit, and run-to-run noise in some phases was
  as large as the size effect.
- **Extrapolation:** the model tells the reader roughly what a larger job
  would take. Beyond the sizes it was fitted on, that is an extrapolation,
  and the pages say so.

---

## 14. How the result pages report numbers

Proportions (accuracy, pass@1, tool-use score, precision, agreement).
Counts such as leak-marker matches are reported as a count, a per-prompt
rate and a Garwood interval instead (Sec. 2):

| Field | Meaning |
| --- | --- |
| n | items scored |
| x | successes (or failures, for a failure rate) |
| estimate | x / n |
| 95% CI | Clopper-Pearson, unless another method is named |
| selection | seeded random sample (with seed and population size), first n, or the whole set |
| decoding | deterministic, or stochastic (backend default / stated temperature) |
| runs | number of independent runs (usually 1) |

Scores that include time-outs (Sec. 3) additionally give t, the number
of time-outs, and the range (x + t)/n that the score would reach if every
time-out had been correct.

Comparisons additionally give the design (paired or unpaired), the
discordant counts b and c for paired data, the test and its p-value, the
interval for the difference, the family and Holm-adjusted p-value where
a family applies, and every known confounder.

Timings and throughput give the definition of the quantity (per request
or aggregate, and which tokens count), the statistic (median, p90 with
its quantile definition, mean), the number of requests and runs, the
interval where raw data allow one, and the conditions: model, context,
concurrency, engine and image, MTP on or off, warm or cold.

---

## 15. Glossary

- **Aggregate throughput** -- total tokens generated by all concurrent requests per second of wall-clock time (Sec. 9).
- **Agreement rate** -- proportion of items on which a student model's answer equals its teacher's; not correctness (Sec. 10).
- **AUROC** -- probability that a random positive outranks a random negative; 0.5 is chance (Sec. 11).
- **Bootstrap (percentile)** -- resampling the observed units with replacement to approximate the sampling distribution of a statistic; the 2.5th and 97.5th percentiles of the resampled statistic form a 95% interval (Sec. 9).
- **Calibration split** -- data used only to fit calibration and choose thresholds (Sec. 10).
- **Calibration** -- agreement between stated confidence and observed accuracy (Sec. 11).
- **Clopper-Pearson interval** -- exact, conservative confidence interval for a binomial proportion (Sec. 3).
- **Cluster bootstrap** -- bootstrap that resamples whole clusters (all decisions about one text) instead of single observations (Sec. 11).
- **Coefficient of variation (CV)** -- standard deviation divided by the mean; used for run-to-run spread of repeated timings (Sec. 9).
- **Confidence interval (95%)** -- output of a procedure that covers the true value in at least (or about) 95% of repeated experiments (Sec. 3).
- **Confounding** -- a second factor that changed together with the one being compared (Sec. 4.3).
- **Coverage (acceptance rate)** -- fraction of items a thresholded model answers (Sec. 10). Not to be confused with the coverage probability of an interval (Sec. 3).
- **Discordant pairs** -- items on which two paired configurations disagree; the only items that inform McNemar's test (Sec. 4.1).
- **Discrimination** -- how well scores rank correct above incorrect items (Sec. 11).
- **Family-wise error rate** -- probability of at least one false rejection among a family of tests (Sec. 6).
- **Finite-population correction** -- variance reduction when sampling without replacement from a finite population; applies in full only to deterministic outcomes (Sec. 3).
- **First-n subset** -- the first n items in file order; supports claims about those items only (Sec. 1).
- **Garwood interval** -- exact confidence interval for a Poisson count, used for count rates such as marker matches per prompt (Sec. 2).
- **Geometric mean of ratios** -- the average of per-item ratios on the log scale; the natural summary of paired speedups (Sec. 9).
- **Held-out split** -- data used once, at the end, to estimate performance (Sec. 10).
- **Holm procedure** -- step-down multiple-comparison correction controlling the family-wise error rate (Sec. 6).
- **Informative truncation** -- outcomes cut off by a limit (a time-out) and scored as failures, mixing speed with correctness (Sec. 7).
- **Margin (equivalence / non-inferiority)** -- the largest difference declared unimportant, fixed before looking at the data (Sec. 5).
- **McNemar test (exact)** -- paired test of two proportions using only the discordant pairs (Sec. 4.1).
- **Median, p50, p90, p95** -- sample quantiles; definitions differ at small n (Sec. 8).
- **Minimum detectable effect** -- the smallest true difference a design detects with a stated power (Sec. 5).
- **Monte Carlo p-value** -- a p-value estimated from random permutations or flips rather than computed exactly; its smallest possible value is 1/(B + 1) for B draws (Sec. 4.5).
- **Newcombe interval (method 10)** -- hybrid score interval for a difference of proportions, in paired and unpaired versions (Sec. 4).
- **One-sided bound** -- a confidence bound in one direction only; the one-sided 95% lower bound equals the lower end of the two-sided 90% interval (Sec. 3).
- **p-value** -- probability under the null hypothesis of a result at least as extreme as the one observed (Sec. 5).
- **Partial identification** -- reporting the range of values compatible with the data when some outcomes are unknown, e.g. [x/n, (x+t)/n] for t time-outs (Sec. 3).
- **pass@1** -- proportion of problems solved by a single generated sample (Sec. 2).
- **Per-request decode rate** -- tokens one request generated divided by its generation time (Sec. 9).
- **Power** -- probability that a test detects a true effect of a given size (Sec. 5).
- **Precision** -- fraction of accepted answers that are correct; its n is the number accepted (Sec. 10).
- **Pseudo-replication** -- treating repeated measurements within one run as if they were independent runs (Sec. 9).
- **Ratio of sums** -- total over total (e.g. all tokens / all time); weights large items more than a mean of ratios (Sec. 9).
- **Seeded random sample** -- items chosen by a pseudo-random shuffle with a fixed seed: a probability sample, reproducible and unbiased in expectation (Sec. 1).
- **Sign-flip test** -- paired randomization test that flips the signs of per-item differences at random to build the null distribution (Sec. 4.5).
- **Single run** -- one realisation of an experiment; no direct information about run-to-run variation (Sec. 1).
- **Spearman rho** -- a rank correlation: the ordinary correlation of the ranks of two variables; robust to outliers and to nonlinear but monotone relations.
- **Stochastic decoding** -- token generation with temperature > 0; makes each scored answer a random draw (Sec. 2).
- **Stratified bootstrap** -- bootstrap that resamples within each stratum (task) separately; used for composite scores (Sec. 2).
- **Tail latency** -- high quantiles of the latency distribution (Sec. 8).
- **Temperature scaling** -- one-parameter calibration, softmax(logits / T), fitted on a calibration split (Sec. 11).
- **Test-retest noise** -- the disagreement between two runs of the same configuration; the floor below which a difference is not evidence (Sec. 4.4).
- **Threshold (tau)** -- the score above which a model's answer is accepted (Sec. 10).
- **Tolerance check** -- deterministic pass/fail comparison against a fixed numerical tolerance; not a statistical test (Sec. 12).
- **Wilson interval** -- score-test confidence interval for a proportion; approximately nominal coverage (Sec. 3).

---

## 16. References

- Clopper, C. J., and Pearson, E. S. (1934). The use of confidence or fiducial limits illustrated in the case of the binomial. *Biometrika* 26(4), 404-413.
- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association* 22(158), 209-212.
- Agresti, A., and Coull, B. A. (1998). Approximate is better than "exact" for interval estimation of binomial proportions. *The American Statistician* 52(2), 119-126.
- Brown, L. D., Cai, T. T., and DasGupta, A. (2001). Interval estimation for a binomial proportion. *Statistical Science* 16(2), 101-133.
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika* 12(2), 153-157.
- Newcombe, R. G. (1998a). Interval estimation for the difference between independent proportions: comparison of eleven methods. *Statistics in Medicine* 17(8), 873-890.
- Newcombe, R. G. (1998b). Improved confidence intervals for the difference between binomial proportions based on paired data. *Statistics in Medicine* 17(22), 2635-2650.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics* 6(2), 65-70.
- Hanley, J. A., and McNeil, B. J. (1982). The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology* 143(1), 29-36.
- DeLong, E. R., DeLong, D. M., and Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. *Biometrics* 44(3), 837-845.
- Hyndman, R. J., and Fan, Y. (1996). Sample quantiles in statistical packages. *The American Statistician* 50(4), 361-365.
- Efron, B., and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman and Hall.
- Lohr, S. L. (2021). *Sampling: Design and Analysis*, 3rd ed. CRC Press. (Finite-population correction.)
- Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML 2017*.
- Chen, M., et al. (2021). Evaluating large language models trained on code. arXiv:2107.03374. (HumanEval and pass@k.)
- Garwood, F. (1936). Fiducial limits for the Poisson distribution. *Biometrika* 28(3/4), 437-442.
- Connor, R. J. (1987). Sample size for testing differences in proportions for the paired-sample design. *Biometrics* 43(1), 207-211.
- Hurlbert, S. H. (1984). Pseudoreplication and the design of ecological field experiments. *Ecological Monographs* 54(2), 187-211.
- Manski, C. F. (2003). *Partial Identification of Probability Distributions*. Springer.
- Hoeffding, W. (1956). On the distribution of the number of successes in independent trials. *Annals of Mathematical Statistics* 27(3), 713-721.
- Imbens, G. W., and Manski, C. F. (2004). Confidence intervals for partially identified parameters. *Econometrica* 72(6), 1845-1857.
- Schuirmann, D. J. (1987). A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. *Journal of Pharmacokinetics and Biopharmaceutics* 15(6), 657-680.
