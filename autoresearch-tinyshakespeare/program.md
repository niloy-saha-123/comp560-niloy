# Program

Goal:
- improve `val_loss` on Tiny Shakespeare under fixed short budget

Rules:
- modify only `train.py`
- do not modify `prepare.py`
- keep evaluation fixed
- keep runtime short and repeatable

Metric:
- primary: `val_loss`
- secondary: `tokens_per_second`

Good changes:
- learning rate
- weight decay
- dropout
- batch size
- model width/depth
- gradient clipping
- optimizer settings

Bad changes:
- dataset changes
- evaluation changes
- hacks tuned to one batch
