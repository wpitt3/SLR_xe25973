import pandas as pd
from metrics import wer_single
from scipy.stats import ttest_rel

def wer(references, hypotheses):
    total_error, total_ref_len = 0, 0

    for r, h in zip(references, hypotheses):
        res = wer_single(r, str(h))
        total_error += res["num_err"]
        total_ref_len += res["num_ref"]

    wer = (total_error / total_ref_len) * 100
    return wer


def individual_wer(references, hypotheses):
    result = []

    for r, h in zip(references, hypotheses):
        res = wer_single(r, str(h))
        result.append(res["num_err"]/ res["num_ref"])
    return result


def calculate_heads(df):
    heads = ["body_gls_hyp", "left_gls_hyp", "right_gls_hyp", "fuse_gls_hyp", "ensemble_last_gls_hyp"]
    for h in heads:
        w = wer(list(df.ref), list(df[h]))
        print(f"{h:30s} {w:.2f}")

NAME_TO_TITLE = {'baseline': 'Baseline', 'arm3': 'Masked High LR', 'arm2': 'Masked Low LR', 'jitter': 'Jitter Low LR'}
MODELS = {"baseline":"preds_baseline.csv", "arm2":"preds_arm2.csv", "arm3":"preds_arm3.csv", "jitter":"preds_jitter.csv"}
HEAD = 'fuse_gls_hyp'

def main():
    calculate_heads(pd.read_csv(f"../logs/preds/preds_baseline.csv"))
    merged = None
    for name, filename in MODELS.items():
        df = pd.read_csv(f"../logs/preds/{filename}")[["name", "ref", HEAD]].copy()
        df[f"wer_{name}"] =individual_wer(list(df.ref), list(df[HEAD]))
        df = df.rename(columns={HEAD: f"hyp_{name}"})
        merged = df if merged is None else merged.merge(df.drop(columns="ref"), on="name")
    corr = merged[[f"wer_{m}" for m in MODELS]].corr()

    model_names = list(MODELS.keys())
    model_names.pop(0)
    clip_count = merged.wer_baseline.count()
    model_predictions = [merged.wer_arm2, merged.wer_arm3, merged.wer_jitter]
    corr = [f"{corr.iloc[0, i+1]:.2f}" for i in range(3)]
    base_worse = [f"{(merged.wer_baseline - x > 0.05).sum()/clip_count:.2f}" for x in model_predictions]
    model_worse = [f"{(x - merged.wer_baseline > 0.05).sum()/clip_count:.2f}" for x in model_predictions]
    t_test = [f"{ttest_rel(merged.wer_baseline, x)[1]:.3f}" for x in model_predictions]


    print(corr)
    print(base_worse)
    print(model_worse)
    print(t_test)
    for a, b, c in (model_names, corr, base_worse, model_worse, t_test):
        print(f" & {a} &  {b} & {c} \\\\")


if __name__ == "__main__":
    main()
