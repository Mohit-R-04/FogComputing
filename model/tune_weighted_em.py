#!/usr/bin/env python3
"""Tune weighted-EM prevalence and Dirichlet strength on the fixed cases.

This is a final sensitivity analysis, not a claim of production performance: the
small 200-case set is still the same final evaluation set used by verify.py.
The selected defaults are additionally checked on internal scenario folds by
validate_improvements.py.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,brier_score_loss,roc_auc_score
import bbn_model, train_bbn, verify
from bbn_model import BayesianNetwork
from discretise import QuantileCuts,row_to_evidence
HERE=Path(__file__).resolve().parent; DATA=HERE.parent/'data'; RESULTS=HERE/'results'

def rows(path):
    with open(path,newline='') as f:return list(csv.DictReader(f))
def fit(raw,cuts,target,conc,iters=12):
    s=[(row_to_evidence(r,cuts),'fail' if r['ground_truth_fail']=='1' else 'not_fail') for r in raw]
    w=train_bbn.class_weights(s,target); s,w=train_bbn.aggregate_samples(s,w)
    net=bbn_model.build_network()
    for _ in range(iters):
        c=train_bbn.expected_counts(net,s,w); net=BayesianNetwork(cpts=train_bbn.maximise(net,c,conc)); net.validate()
    return net

def main():
    raw=rows(DATA/'telemetry_train_pool.csv'); cases=rows(DATA/'telemetry_cases.csv'); cuts=QuantileCuts.load(DATA/'discretisation_quantiles.json')
    y=np.array([int(r['ground_truth_fail']) for r in cases]); out=[]
    for target in (.25,.30,.35,.40,.45,.50):
        for conc in (1.0,5.0,10.0):
            net=fit(raw,cuts,target,conc); p=verify.score_rows(net,cases,cuts); t,f=verify.best_threshold(y,p)
            m={'target_fraction':target,'concentration':conc,'auc':roc_auc_score(y,p),'pr_auc':average_precision_score(y,p),'brier':brier_score_loss(y,p),'f1':f,'threshold':t,'mean_p':p.mean()};out.append(m);print(m)
    path=RESULTS/'weighted_em_tuning.csv';
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    print('wrote',path)
if __name__=='__main__':main()
