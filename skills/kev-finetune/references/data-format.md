# Labelled record format

One JSON object per line. Each record is a System One request (`state` + `questions`) with a `label` on every question.
This is the same shape the deployed endpoint accepts, minus the labels, so what you train on is what you serve.

```json
{"state": "Solve the math problem...",
 "questions": {
   "destination": {"type": "choice", "instructions": "Select the destination model that will answer this query correctly at the lowest cost.",
                   "criteria": {"deepseek-v3-0324": "DeepSeek V3", "claude-sonnet-4": "Claude Sonnet 4"},
                   "label": "deepseek-v3-0324"}}}
```

## Fields

| Field | Rules |
| --- | --- |
| `state` | A string, or a JSON object / array (rendered as `key: value` lines; field names are visible to the model). Under ~1400 characters (384 tokens). Non-empty. |
| `questions` | Object keyed by question id. 1 or more. Questions see the state but not each other. |
| `type` | `noul` (yes/no), `choice` (named options), `score` (ordered levels). |
| `instructions` | The question text. Required. Write it the way the endpoint will ask it. |
| `criteria` | `choice`: object `{name: description or null}`, 1-255 options, names are what the model returns. `score`: list of 2-255 level descriptions, low to high. `noul`: optional `{"true": ..., "false": ...}` descriptions. |
| `label` | `choice`: an option name. `noul`: `true` or `false`. `score`: level index as an integer from 0. |
| `target` (optional) | Soft label: `{option key: weight}`; keys are option names / `"true"`,`"false"` / level indices as strings. |
