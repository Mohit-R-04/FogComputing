#!/usr/bin/env python3
"""
The closed-loop fault-tolerance experiment, inside iFogSim.

    python run_ft_experiments.py           # full experiment (450 runs, about 15 minutes)
    python run_ft_experiments.py --quick   # one held-out run per policy

Every run simulates one held-out scenario with one unseen seed under one policy (5
held-out scenarios x 5 seeds, ``design.FT_SEEDS``). The
trained BBN (``data/bbn_trained.json``), the exported best classical model
(``data/models/ml_best.json``) or the FTAPA threshold (``data/models/ftapa.json``) is
loaded by the Java controller and makes the decisions every epoch, while the
degradation fault model fails nodes and the failures genuinely stop their service. The
oracle policy runs the same controller as the BBN and ML policies with the best possible
predictor (it knows the hidden health and the wear law, not future random draws): the
most any predictor could achieve.

Policies see identical exogenous randomness -- the same initial wear, wear noise, shocks
and sudden failures, keyed on (seed, scenario, node, epoch) -- so differences between
policies are caused by their decisions alone. They can still change how many failures
happen: a policy that keeps users off fragile nodes slows their wear.

Experiments:

  main         all seven policies, complete telemetry
  degraded     the telemetry-driven policies with 30% of channels missing, more often
               on degrading nodes (MNAR)
  sensitivity  the BBN policy with every migration cost scaled by 0.25 ... 4
  ablation     the BBN policy with one component removed at a time: online CPT learning,
               the risk term of placement (Eq. 2), proactive migration (Eq. 3), and
               selective replication (Eq. 4)

Outputs under model/results/ft/: one JSON per run (runs/), summary.json,
policy_comparison.csv/.md and the figures.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import design
import javarun
import vizstyle

OUT = design.RESULTS / "ft"
RUNS = OUT / "runs"
POLICIES = ["no-ft", "cloud-only", "reactive", "threshold", "ml", "bbn", "oracle"]
TELEMETRY_POLICIES = ["threshold", "ml", "bbn"]
LABELS = {
    "no-ft": "No fault tolerance",
    "cloud-only": "Cloud-only placement",
    "reactive": "Reactive checkpoint + backup (Rajab 2021)",
    "threshold": "Threshold migration, FTAPA (Gupta 2025)",
    "ml": "Expected-loss controller + ML",
    "bbn": "Proposed BBN framework",
    "oracle": "Oracle predictor (upper bound)",
}
COST_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)
ABLATIONS = {
    "offline": ("--no-online", "without online CPT learning"),
    "norisk": ("--no-risk-placement", "without the risk term in placement (N2)"),
    "nomig": ("--no-migration", "without proactive migration (N2)"),
    "norep": ("--no-replication", "without selective replication (N3)"),
}
DEGRADED_RATE = 0.30


def run_name(policy: str, scenario: int, seed: int, tag: str = "") -> str:
    return f"{policy}_s{scenario:02d}_seed{seed}{('_' + tag) if tag else ''}"


def job(policy: str, scenario: int, seed: int, tag: str = "", extra=()):
    out = RUNS / f"{run_name(policy, scenario, seed, tag)}.json"
    args = ["--policy", policy, "--scenario", str(scenario), "--seed", str(seed),
            "--out", javarun.relpath_from_ifogsim(out), "--no-summary",
            "--bbn", javarun.relpath_from_ifogsim(design.DATA / "bbn_trained.json"),
            "--ml", javarun.relpath_from_ifogsim(design.MODELS / "ml_best.json"),
            "--ftapa", javarun.relpath_from_ifogsim(design.MODELS / "ftapa.json"),
            "--sim-time", str(design.SIM_TIME), *extra]
    return ("FtExperiment", args), out


def plan(quick: bool):
    pairs = [(s, k) for s in design.TEST_SCENARIOS for k in design.FT_SEEDS]
    if quick:
        pairs = pairs[:1]
    jobs = []
    for s, k in pairs:
        for p in POLICIES:
            jobs.append(("main", p, s, k, "", ()))
        if not quick:
            for p in TELEMETRY_POLICIES:
                jobs.append(("degraded", p, s, k, "mnar30",
                             ("--dropout", "mnar", "--rate", str(DEGRADED_RATE))))
            for c in COST_SCALES:
                if c != 1.0:
                    jobs.append(("sensitivity", "bbn", s, k, f"cost{c:g}", ("--cost-scale", str(c))))
            for tag, (flag, _) in ABLATIONS.items():
                jobs.append(("ablation", "bbn", s, k, tag, (flag,)))
    return jobs


def load(path: Path, experiment: str, tag: str) -> dict:
    d = json.loads(path.read_text())
    d["experiment"] = experiment
    d["tag"] = tag
    return d


def flatten(d: dict) -> dict:
    keys = ["experiment", "tag", "policy", "scenario", "seed", "users", "fog_failures",
            "fog_failures_with_users", "sessions_hit", "sessions_evacuated", "edge_failures",
            "downtime_total", "downtime_failure", "downtime_migration", "downtime_handover",
            "availability", "delivery_ratio", "loop_delay_mean", "loop_delay_p95",
            "critical_deadline_miss_ratio", "migrations_started", "migrations_completed",
            "migrations_aborted", "migrations_unnecessary", "migrations_from_degraded", "handovers",
            "standby_ratio",
            "replication_mb", "lead_time_mean", "energy_cloud", "energy_fog", "energy_edge",
            "network_usage", "controller_ms_per_epoch", "dropout", "dropout_rate"]
    row = {k: d.get(k) for k in keys}
    row["critical_downtime"] = d["by_class"]["critical"]["downtime"]
    row["critical_availability"] = d["by_class"]["critical"]["availability"]
    exposures = (d["sessions_hit"] or 0) + (d["sessions_evacuated"] or 0)
    row["exposures_avoided"] = (d["sessions_evacuated"] / exposures) if exposures else np.nan
    row["downtime_per_user"] = d["downtime_total"] / max(1, d["users"])
    # What fault tolerance controls: failures (detection + recovery) and migration pauses.
    # Handover downtime comes from users walking between gateways.
    row["fault_downtime"] = d["downtime_failure"] + d["downtime_migration"]
    row["fault_downtime_per_user"] = row["fault_downtime"] / max(1, d["users"])
    row["handover_downtime_per_user"] = d["downtime_handover"] / max(1, d["users"])
    by_cause = d.get("sessions_hit_by_cause") or {}
    row["hit_sudden"] = by_cause.get("sudden", 0)
    row["hit_foreseeable"] = sum(v for k, v in by_cause.items() if k != "sudden")
    return row


def paired_test(df: pd.DataFrame, a: str, b: str, metric: str) -> dict:
    x = df[df.policy == a].set_index(["scenario", "seed"])[metric]
    y = df[df.policy == b].set_index(["scenario", "seed"])[metric]
    both = pd.concat([x, y], axis=1, keys=["a", "b"]).dropna()
    if len(both) < 3 or np.allclose(both.a, both.b):
        return {"n": len(both), "p_value": float("nan"),
                "mean_change_pct": 0.0 if len(both) else float("nan")}
    stat = wilcoxon(both.a, both.b)
    change = 100.0 * (both.a.mean() - both.b.mean()) / max(1e-9, abs(both.b.mean()))
    return {"n": len(both), "p_value": float(stat.pvalue), "mean_change_pct": float(change)}


# ---- figures ---------------------------------------------------------------------------------

def figures(main: pd.DataFrame, degraded: pd.DataFrame, sens: pd.DataFrame, timeline: dict,
            timeline_run: tuple[int, int] | None = None) -> list[str]:
    import matplotlib.pyplot as plt
    vizstyle.apply()
    names = []

    # Downtime per policy: the part fault tolerance controls (failures + migration pauses)
    # and the part mobility causes (handovers), with the degraded-telemetry marks beside.
    fault = main.groupby("policy").fault_downtime_per_user.mean().reindex(POLICIES)
    hand = main.groupby("policy").handover_downtime_per_user.mean().reindex(POLICIES)
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    x = np.arange(len(POLICIES))
    colors = [vizstyle.POLICY_COLORS["bbn"] if p == "bbn" else vizstyle.INK_2 for p in POLICIES]
    ax.bar(x, fault, width=0.55, color=colors, zorder=2, label="failures + migration pauses")
    ax.bar(x, hand, width=0.55, bottom=fault, color=vizstyle.RECEDE, zorder=2,
           label="mobility handovers (not a fault-tolerance decision)")
    dg = degraded.groupby("policy").fault_downtime_per_user.mean() if len(degraded) else pd.Series(dtype=float)
    if len(degraded):
        for i, p in enumerate(POLICIES):
            if p in dg.index:
                ax.plot([i - 0.3, i + 0.3], [dg[p], dg[p]], color=vizstyle.SLOTS[1], lw=2.2, zorder=4)
        ax.plot([], [], color=vizstyle.SLOTS[1], lw=2.2, label="failures + migrations with 30% telemetry loss (MNAR)")
    for i, p in enumerate(POLICIES):
        top = max(fault.iloc[i] + hand.iloc[i], dg.get(p, 0.0))
        ax.text(i, top + 6, f"{fault.iloc[i]:.0f}", ha="center", fontsize=8.5,
                color=vizstyle.INK, fontweight="bold" if p == "bbn" else "normal")
    if "cloud-only" in POLICIES:
        i = POLICIES.index("cloud-only")
        ax.text(i, fault.iloc[i] + hand.iloc[i] + 30, "misses every\ncritical deadline", ha="center",
                fontsize=7.5, color=vizstyle.MUTED)
    ax.set_xticks(x, ["\n".join(textwrap.wrap(LABELS[p], 15)) for p in POLICIES], fontsize=8)
    ax.set_ylabel("Service downtime per user (ms of simulated time)")
    ax.set_title(f"Service downtime on held-out runs (mean of {main.groupby('policy').size().max()} runs; "
                 "number = failures + migrations)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "ft_downtime.png", dpi=170)
    plt.close(fig)
    names.append("ft_downtime.png")

    # Downtime versus redundancy overhead (Objective 3). Log scale: the policies span
    # two orders of magnitude; labels are spread apart and tied to their points.
    pts = main.groupby("policy")[["fault_downtime_per_user", "standby_ratio", "replication_mb"]].mean().reindex(POLICIES)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    last = None
    for p, r in sorted(pts.iterrows(), key=lambda kv: kv[1].fault_downtime_per_user):
        c = vizstyle.POLICY_COLORS[p]
        xv, yv = 100 * r.standby_ratio, max(r.fault_downtime_per_user, 1.0)
        ty = yv if last is None else max(yv, last * 1.7)
        last = ty
        ax.scatter(xv, yv, s=70, color=c, zorder=3, edgecolor=vizstyle.SURFACE, linewidth=2)
        right = xv < 70
        ax.annotate(LABELS[p], (xv, yv), xytext=(xv + 8 if right else xv - 8, ty), textcoords="data",
                    ha="left" if right else "right", va="center", fontsize=8.5, color=vizstyle.INK_2,
                    fontweight="bold" if p == "bbn" else "normal",
                    arrowprops=dict(arrowstyle="-", color=vizstyle.MUTED, lw=0.6, shrinkA=0, shrinkB=4))
    ax.set(yscale="log", xlabel="Sessions holding a reserved warm standby (%)",
           ylabel="Downtime from failures + migrations per user (ms)",
           title="Fault-related downtime against reserved standby capacity")
    ax.set_xlim(-5, 110)
    ax.set_ylim(1, 1500)
    fig.tight_layout()
    fig.savefig(OUT / "ft_tradeoff.png", dpi=170)
    plt.close(fig)
    names.append("ft_tradeoff.png")

    if len(sens):
        s = sens.groupby("cost_scale")[["fault_downtime_per_user", "migrations_started"]].mean().reset_index()
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        axes[0].plot(s.cost_scale, s.fault_downtime_per_user, marker="o", color=vizstyle.SLOTS[0])
        axes[0].set(xscale="log", title="Downtime from failures + migrations", xlabel="Migration cost multiplier",
                    ylabel="Downtime per user (ms)")
        axes[1].plot(s.cost_scale, s.migrations_started, marker="o", color=vizstyle.SLOTS[0])
        axes[1].set(xscale="log", title="Proactive migrations", xlabel="Migration cost multiplier",
                    ylabel="Migrations per run")
        axes[0].set_ylim(0, max(s.fault_downtime_per_user) * 1.4)
        axes[1].set_ylim(0, max(s.migrations_started) * 1.15)
        for ax in axes:
            ax.minorticks_off()
            ax.set_xticks(list(COST_SCALES), [f"{c:g}" for c in COST_SCALES])
        fig.suptitle("BBN policy: sensitivity to the cost ratio C_mig / C_fail", x=0.01, ha="left",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUT / "ft_sensitivity.png", dpi=170)
        plt.close(fig)
        names.append("ft_sensitivity.png")

    if timeline:
        fig, ax = plt.subplots(figsize=(9, 4.6))
        for p in POLICIES[::-1]:
            if p not in timeline:
                continue
            t = [e["t"] for e in timeline[p]]
            d = [e["downtime"] / 1000.0 for e in timeline[p]]
            ax.plot(t, d, color=vizstyle.POLICY_COLORS[p], lw=2.4 if p == "bbn" else 1.5, label=LABELS[p])
        where = f" (scenario {timeline_run[0]}, seed {timeline_run[1]})" if timeline_run else ""
        ax.set(title=f"Cumulative downtime through the held-out run with the most fog failures{where}",
               xlabel="Simulation time", ylabel="Cumulative downtime (s, all users, incl. handovers)")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], fontsize=8, loc="upper left")
        fig.tight_layout()
        fig.savefig(OUT / "ft_timeline.png", dpi=170)
        plt.close(fig)
        names.append("ft_timeline.png")
    return names


# ---- report ------------------------------------------------------------------------------------

def report(main: pd.DataFrame, degraded: pd.DataFrame, sens: pd.DataFrame, ablation: pd.DataFrame,
           tests: dict, degraded_tests: dict, figs: list[str]) -> None:
    def ms(series, digits=1, pct=False):
        m, s = series.mean(), series.std()
        if pct:
            return f"{100 * m:.{digits}f} ± {100 * s:.{digits}f}"
        return f"{m:.{digits}f} ± {s:.{digits}f}"

    def paired(a: pd.DataFrame, b: pd.DataFrame, metric: str) -> float:
        both = pd.concat([a.set_index(["scenario", "seed"])[metric], b.set_index(["scenario", "seed"])[metric]],
                         axis=1, keys=["a", "b"]).dropna()
        if len(both) < 3 or np.allclose(both.a, both.b):
            return float("nan")
        return float(wilcoxon(both.a, both.b).pvalue)

    def pv(p: float) -> str:
        return "n/a" if p != p else ("<0.0001" if p < 1e-4 else f"{p:.4f}")

    n_runs = main.groupby("policy").size().max()
    L = ["# Closed-loop fault tolerance in iFogSim", "",
         f"Each policy ran on the same {n_runs} held-out runs ({len(design.TEST_SCENARIOS)} unseen scenarios x "
         f"{len(design.FT_SEEDS)} unseen seeds, "
         f"{design.SIM_TIME} time units each) with identical exogenous randomness. Values are mean ± "
         "standard deviation over runs.", "",
         "Downtime has three sources: node failures (detection plus recovery), the pauses of proactive "
         "migrations, and mobility handovers (a user walking to another gateway is re-associated, whatever "
         "the policy). Fault tolerance acts on the first two, reported as *failures + migrations*; "
         "handover downtime is about the same for every policy that serves users from the fog.", "",
         "| Policy | Downtime / user (ms) | Failures + migrations / user (ms) | Availability (%) | Critical availability (%) | Alarms delivered (%) | Fog failures | Users hit | Users evacuated | Migrations (from degraded nodes) | Warm standby (%) | Loop p95 (ms) | Critical deadline misses (%) |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for p in POLICIES:
        g = main[main.policy == p]
        if g.empty:
            continue
        name = f"**{LABELS[p]}**" if p == "bbn" else LABELS[p]
        L.append(f"| {name} | {ms(g.downtime_per_user)} | {ms(g.fault_downtime_per_user)} | "
                 f"{ms(g.availability, 3, True)} | {ms(g.critical_availability, 3, True)} | "
                 f"{ms(g.delivery_ratio, 2, True)} | {ms(g.fog_failures)} | {ms(g.sessions_hit)} | "
                 f"{ms(g.sessions_evacuated)} | {g.migrations_started.mean():.1f} "
                 f"({g.migrations_from_degraded.mean():.1f}) | {100 * g.standby_ratio.mean():.1f} | "
                 f"{g.loop_delay_p95.mean():.1f} | {100 * g.critical_deadline_miss_ratio.mean():.2f} |")
    L += ["", "*Users hit*: sessions on a node when it failed. *Users evacuated*: sessions moved off a node "
          "within the horizon before it failed. *From degraded nodes*: migrations whose source node was "
          "genuinely worn (hidden health < 0.42, i.e. at least 1.5x work inflation) when the decision was "
          "made -- a migration that unloads such a node can prevent its failure, so 'the node did not fail "
          "afterwards' is not evidence of a false alarm. The *oracle* runs the same controller as the BBN "
          "and ML policies with the best possible predictor: it knows each node's hidden health and the "
          "wear law (not future random draws, so sudden failures still surprise it), which bounds what any "
          "predictor could achieve. *Cloud-only* avoids "
          "fog state loss but every alarm crosses the WAN: it misses the 150 ms critical deadline on every "
          "loop.", ""]
    L += ["Users hit, by how their node failed (sudden failures have no precursor by construction):", "",
          "| Policy | Users hit by wear-out or shock failures | Users hit by sudden failures |", "|---|---:|---:|"]
    for p in POLICIES:
        g = main[main.policy == p]
        if not g.empty:
            name = f"**{LABELS[p]}**" if p == "bbn" else LABELS[p]
            L.append(f"| {name} | {g.hit_foreseeable.mean():.1f} | {g.hit_sudden.mean():.1f} |")
    L.append("")
    for f in figs:
        L += [f"![{f}]({f})", ""]
    L += ["## Paired comparisons (BBN against each policy)", "",
          "Wilcoxon signed-rank test over the paired held-out runs; negative change means the BBN "
          "framework has less downtime.", "",
          "| Against | Downtime change (%) | p-value | Failures + migrations change (%) | p-value | Critical downtime change (%) | p-value |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for p, t in tests.items():
        L.append(f"| {LABELS[p]} | {t['downtime']['mean_change_pct']:+.1f} | {pv(t['downtime']['p_value'])} | "
                 f"{t['fault']['mean_change_pct']:+.1f} | {pv(t['fault']['p_value'])} | "
                 f"{t['critical']['mean_change_pct']:+.1f} | {pv(t['critical']['p_value'])} |")
    if len(degraded):
        L += ["", "## With degraded telemetry (30% of channels missing, MNAR)", "",
              "Channels go missing more often on degrading nodes (rate x (1 + damage)), the case Review 1 "
              "raises: a channel falls silent exactly when its node is in trouble.", "",
              "| Policy | Downtime / user (ms) | Failures + migrations / user (ms) | Change vs complete telemetry (%) | Users hit | Users evacuated |",
              "|---|---:|---:|---:|---:|---:|"]
        for p in TELEMETRY_POLICIES:
            g = degraded[degraded.policy == p]
            base = main[main.policy == p].fault_downtime_per_user.mean()
            if g.empty:
                continue
            name = f"**{LABELS[p]}**" if p == "bbn" else LABELS[p]
            L.append(f"| {name} | {ms(g.downtime_per_user)} | {ms(g.fault_downtime_per_user)} | "
                     f"{100 * (g.fault_downtime_per_user.mean() - base) / max(1e-9, base):+.1f} | "
                     f"{ms(g.sessions_hit)} | {ms(g.sessions_evacuated)} |")
        if degraded_tests:
            L += ["", "| BBN against, with degraded telemetry | Failures + migrations change (%) | p-value | Downtime change (%) | p-value |",
                  "|---|---:|---:|---:|---:|"]
            for p, t in degraded_tests.items():
                L.append(f"| {LABELS[p]} | {t['fault']['mean_change_pct']:+.1f} | {pv(t['fault']['p_value'])} | "
                         f"{t['downtime']['mean_change_pct']:+.1f} | {pv(t['downtime']['p_value'])} |")
    if len(sens):
        L += ["", "## Sensitivity of the BBN policy to the cost ratio", "",
              "Every migration cost multiplied by the factor shown (the derived threshold "
              "P* = C_mig / C_fail scales with it).", "",
              "| Cost multiplier | Failures + migrations / user (ms) | Migrations (from degraded nodes) | Users hit | Users evacuated | Warm standby (%) |",
              "|---:|---:|---:|---:|---:|---:|"]
        for c, g in sens.groupby("cost_scale"):
            L.append(f"| {c:g} | {ms(g.fault_downtime_per_user)} | {g.migrations_started.mean():.1f} "
                     f"({g.migrations_from_degraded.mean():.1f}) | {g.sessions_hit.mean():.1f} | "
                     f"{g.sessions_evacuated.mean():.1f} | {100 * g.standby_ratio.mean():.1f} |")
    if len(ablation):
        full = main[main.policy == "bbn"]
        base = full.fault_downtime_per_user.mean()
        L += ["", "## Ablation: what each component contributes", "",
              "The BBN policy on the same runs with one component removed. The p-value is a paired Wilcoxon "
              "test on failures + migrations downtime.", "",
              "| Variant | Failures + migrations / user (ms) | Change vs full (%) | p-value | Fog failures | Users hit | Users evacuated | Migrations | Warm standby (%) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        L.append(f"| full framework | {ms(full.fault_downtime_per_user)} | -- | | {full.fog_failures.mean():.1f} | "
                 f"{full.sessions_hit.mean():.1f} | {full.sessions_evacuated.mean():.1f} | "
                 f"{full.migrations_started.mean():.1f} | {100 * full.standby_ratio.mean():.1f} |")
        for tag, (_, label) in ABLATIONS.items():
            g = ablation[ablation.tag == tag]
            if g.empty:
                continue
            L.append(f"| {label} | {ms(g.fault_downtime_per_user)} | "
                     f"{100 * (g.fault_downtime_per_user.mean() - base) / max(1e-9, base):+.1f} | "
                     f"{pv(paired(full, g, 'fault_downtime'))} | {g.fog_failures.mean():.1f} | "
                     f"{g.sessions_hit.mean():.1f} | {g.sessions_evacuated.mean():.1f} | "
                     f"{g.migrations_started.mean():.1f} | {100 * g.standby_ratio.mean():.1f} |")
    (OUT / "policy_comparison.md").write_text("\n".join(L) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--reuse", action="store_true", help="aggregate existing run files only")
    args = ap.parse_args(argv)
    for f in (design.DATA / "bbn_trained.json", design.MODELS / "ml_best.json", design.MODELS / "ftapa.json"):
        if not f.exists():
            sys.exit(f"missing {f}; run train_bbn.py and comparison/run_comparison.py first")
    javarun.ensure_built()
    RUNS.mkdir(parents=True, exist_ok=True)

    jobs = plan(args.quick)
    specs = [(exp, p, s, k, tag, *job(p, s, k, tag, extra)) for exp, p, s, k, tag, extra in jobs]
    start = time.time()
    if not args.reuse:
        def progress(done, total, res):
            if done % 10 == 0 or done == total:
                print(f"  {done}/{total} runs finished ({time.time() - start:.0f} s)")
        javarun.run_many([sp[5] for sp in specs], workers=args.workers, on_done=progress)

    rows, timelines = [], {}
    for exp, p, s, k, tag, _, out in specs:
        d = load(out, exp, tag)
        if exp == "main":
            timelines[(s, k, p)] = d["timeline"]
        r = flatten(d)
        if exp == "sensitivity":
            r["cost_scale"] = float(tag.replace("cost", ""))
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "policy_runs.csv", index=False)
    main_df = df[df.experiment == "main"].copy()
    degraded = df[df.experiment == "degraded"].copy()
    sens = pd.concat([df[df.experiment == "sensitivity"],
                      main_df[main_df.policy == "bbn"].assign(cost_scale=1.0)]) if (df.experiment == "sensitivity").any() else pd.DataFrame()
    ablation = df[df.experiment == "ablation"].copy()

    tests = {}
    for p in POLICIES:
        if p == "bbn":
            continue
        tests[p] = {"downtime": paired_test(main_df, "bbn", p, "downtime_total"),
                    "fault": paired_test(main_df, "bbn", p, "fault_downtime"),
                    "critical": paired_test(main_df, "bbn", p, "critical_downtime")}
    degraded_tests = {}
    for p in TELEMETRY_POLICIES:
        if p != "bbn" and (degraded.policy == p).any() and (degraded.policy == "bbn").any():
            degraded_tests[p] = {"downtime": paired_test(degraded, "bbn", p, "downtime_total"),
                                 "fault": paired_test(degraded, "bbn", p, "fault_downtime")}
    # Illustration: the held-out run where failures matter most (most fog failures without
    # fault tolerance); the aggregate results above cover every run.
    nf = main_df[main_df.policy == "no-ft"]
    timeline_run = tuple(int(v) for v in nf.loc[nf.fog_failures.idxmax(), ["scenario", "seed"]]) if len(nf) else None
    timeline = {p: timelines[(*timeline_run, p)] for p in POLICIES if timeline_run and (*timeline_run, p) in timelines}
    figs = figures(main_df, degraded, sens, timeline, timeline_run)
    report(main_df, degraded, sens, ablation, tests, degraded_tests, figs)
    summary = {
        "runs": int(len(df)),
        "seconds": round(time.time() - start, 1),
        "policies": {p: main_df[main_df.policy == p].drop(columns=["experiment", "tag", "dropout"]).mean(numeric_only=True).to_dict()
                     for p in POLICIES if (main_df.policy == p).any()},
        "degraded": {p: degraded[degraded.policy == p].mean(numeric_only=True).to_dict()
                     for p in TELEMETRY_POLICIES if (degraded.policy == p).any()},
        "tests": tests,
        "degraded_tests": degraded_tests,
        "sensitivity": sens.groupby("cost_scale")[["downtime_per_user", "fault_downtime_per_user", "migrations_started"]].mean().reset_index().to_dict(orient="records") if len(sens) else [],
        "ablations": {tag: ablation[ablation.tag == tag].mean(numeric_only=True).to_dict()
                      for tag in ABLATIONS if (ablation.tag == tag).any()},
        "labels": LABELS,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"closed-loop experiment: {len(df)} runs aggregated in {time.time() - start:.0f} s -> "
          f"{OUT / 'policy_comparison.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
