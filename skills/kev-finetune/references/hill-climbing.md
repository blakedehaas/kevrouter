# Reading the numbers and improving the model

`train` and `evaluate` write `runs/<name>/result.json` and print a table. Every metric is computed over development
*questions* (one record with three questions is three rows) by `kev.metrics.metrics`.

## The Metrics Table

| Metric | Meaning | What to want |
| --- | --- | --- |
| accuracy | argmax == label | up; temperature never changes it |
| brier | sum of squared probability error | down; rewards being right *and* honest |
| nll | negative log-likelihood of the label | down; the temperature is fitted to minimize this on calibration |
| ece | expected calibration error (10 bins) | down; calibrated should be well under raw |
| confident errors (p>=0.9, wrong) | share of *all* questions answered wrong with >= 0.9 | down, ideally near 0: these are the silent failures |
| coverage at 5% error | largest share of questions that can be accepted, highest confidence first, with <= 5% error among them | up: how much of the workload can be automated |
| mean confidence | average top probability | should track accuracy; far above accuracy = overconfident |

Columns come in pairs: `raw` is the model's logits as trained; `calibrated` divides them by the temperature fitted
on `calibration.jsonl`. Both models get their own fit on your calibration slice, so the comparison is fair.

## Deciding

1. **Is the gain real?** `bootstrap` in `result.json` holds paired bootstrap deltas (fine-tuned minus baseline) for
   `acc`, `brier` and `ece` with 95% CIs, resampling development records.
2. **Did it forget?** `regression` scores both models on public development records.
3. **Is it honest?** Calibrated `confident_error_rate` of the fine-tuned model should be <= the baseline's, and
   `mean_conf` within a few points of `acc`.
4. **Per question.** `development.per_question` breaks metrics down by question id.
