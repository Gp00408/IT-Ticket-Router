# Model Evaluation

Both models were evaluated on the same 72 held-out test tickets (the 20% split
in Round 2 that neither model ever saw during training).

## Category model (Round 3)

**Accuracy: 79.2%**

| Category | Precision | Recall | F1   | Support |
| -------- | --------- | ------ | ---- | ------- |
| Access   | 0.882     | 0.833  | 0.857| 18      |
| Hardware | 0.842     | 0.889  | 0.865| 18      |
| Network  | 0.722     | 0.722  | 0.722| 18      |
| Software | 0.722     | 0.722  | 0.722| 18      |

Confusion matrix (rows = true, columns = predicted):

```
             Access  Hardware  Network  Software
Access           15         0        0         3
Hardware          0        16        2         0
Network            1         2       13         2
Software          1         1        3        13
```

Network and Software are the weakest categories and get confused with each
other most often — makes sense, since tickets about slow tools or dropped
connections can use similar vocabulary regardless of which system is
actually at fault.

## Urgency model (Round 4)

**Accuracy: 81.9%**

| Urgency | Precision | Recall | F1   | Support |
| ------- | --------- | ------ | ---- | ------- |
| Low     | 0.864     | 0.792  | 0.826| 24      |
| Medium  | 0.720     | 0.750  | 0.735| 24      |
| High    | 0.880     | 0.917  | 0.898| 24      |

Confusion matrix (rows = true, columns = predicted):

```
           Low  Medium  High
Low         19       5     0
Medium       3      18     3
High         0       2    22
```

High urgency is the model's strongest class, and mistakes stay "next door"
(Low/Medium or Medium/High) rather than jumping from Low straight to High —
a reasonable failure mode for an urgency triage tool.

## Known limitation

Both models are TF-IDF + Logistic Regression, which means they match on
literal words and short phrases seen during training, not actual meaning.
A ticket phrased very differently from anything in the training set can be
misclassified even if a human would find it obvious — e.g. "my classes
start in an hour" wasn't recognized as high urgency the way "I have a
client presentation in an hour" was, because the wording doesn't overlap
enough for the model to connect them. This is a normal limitation of this
modeling approach on a small (360-ticket) training set, not a bug.
