#!/usr/bin/env python3
"""Choose improvement settings using training scenarios only.

The final evaluation scenarios from dataset_manifest.json are never used here.
Two internal scenario holdouts are evaluated, and the preferred setting is chosen
by mean AUC/Brier across them.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import bbn_model
import build_dataset
import train_bbn
import verify
from discretise import QuantileCuts, row_to_evidence
from bbn_model import BayesianNetwork

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RESULTS = HERE / "results"


def load_raw():
    rows=[]
    for p in sorted((DATA / "raw").glob("scenario_*.csv")):
        with open(p, newline="") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def sample_balanced(rows, n=2000, seed=999):
    rng=random.Random(seed)
    pos=[r for r in rows if r["ground_truth_fail"]=="1"]
    neg=[r for r in rows if r["ground_truth_fail"]=="0"]
    return [rng.choice(pos) for _ in range(round(n*.4))] + [rng.choice(neg) for _ in range(n-round(n*.4))]


def fit(rows, cuts, iterations, concentration):
    samples=[(row_to_evidence(r,cuts), "fail" if r["ground_truth_fail"]=="1" else "not_fail") for r in rows]
    net=bbn_model.build_network()
    for _ in range(iterations):
        counts=train_bbn.expected_counts(net,samples)
        net=BayesianNetwork(cpts=train_bbn.maximise(net,counts,concentration))
    net.validate()
    return net


def eval_model(net, rows, cuts):
    y=np.array([int(r["ground_truth_fail"]) for r in rows])
    p=verify.score_rows(net, rows, cuts)
    return {"auc":float(roc_auc_score(y,p)), "pr_auc":float(average_precision_score(y,p)),
            "brier":float(brier_score_loss(y,p))}


def main():
    manifest=json.loads((DATA/"dataset_manifest.json").read_text())
    all_rows=load_raw()
    train_ids=set(manifest["scenarios_train"])
    # Separate internal validation holdouts from the scenarios used by the final model.
    folds=([5,11],[0,4])
    candidates=[("baseline",8,20.0,False), ("balanced_c1",12,1.0,True),
                ("balanced_c5",12,5.0,True), ("balanced_c10",12,10.0,True)]
    output=[]
    for fold in folds:
        val_ids=set(fold)
        fit_rows=[r for r in all_rows if int(r["scenario_id"]) in train_ids-val_ids]
        val_rows=[r for r in all_rows if int(r["scenario_id"]) in val_ids]
        cuts=QuantileCuts.load(DATA/"discretisation_quantiles.json") if not fit_rows else QuantileCuts(
            global_cuts={k:tuple(v) for k,v in []}, by_class={})
        # Compute cuts on fit rows only; this prevents validation distribution leakage.
        q=build_dataset.build_quantiles(fit_rows)
        def parse(block): return {k:(float(v["q33"]),float(v["q67"])) for k,v in block.items()}
        cuts=QuantileCuts(global_cuts=parse(q["global"]), by_class={c:parse(b) for c,b in q["by_device_class"].items()}, n_snapshots=len(fit_rows))
        for name,it,conc,balanced in candidates:
            train_rows=sample_balanced(fit_rows,2000,seed=999) if balanced else sample_balanced(fit_rows,2000,seed=999)
            if not balanced:
                # Match the current builder's sampling: use all available positives first,
                # then negatives, so this is the current baseline rather than a resample.
                rng=random.Random(999)
                pos=[r for r in fit_rows if r["ground_truth_fail"]=="1"]
                neg=[r for r in fit_rows if r["ground_truth_fail"]=="0"]
                rng.shuffle(pos); rng.shuffle(neg)
                train_rows=pos[:min(round(2000*.4),len(pos))]+neg[:2000-min(round(2000*.4),len(pos))]
            net=fit(train_rows,cuts,it,conc)
            m=eval_model(net,val_rows,cuts)
            m.update({"fold":"-".join(map(str,fold)),"name":name,"n_train":len(train_rows),"positive_train":sum(r["ground_truth_fail"]=="1" for r in train_rows),"iterations":it,"concentration":conc})
            output.append(m)
            print(m)
    out=RESULTS/"improvement_validation.csv"
    with open(out,"w",newline="") as fh:
        fields=["fold","name","n_train","positive_train","iterations","concentration","auc","pr_auc","brier"]
        w=csv.DictWriter(fh,fieldnames=fields); w.writeheader(); w.writerows(output)
    print(f"wrote {out}")

if __name__=="__main__": main()
