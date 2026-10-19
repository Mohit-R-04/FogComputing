# -*- coding: utf-8 -*-
"""Content for Review2_ResearchPaper.docx.

All prose, tables, formulas and references live here as data; build_paper.py turns
them into the .docx.  Citations are written as placeholders and expanded by cite():

    {key}      -> "(Rajab and Younis 2021)"        parenthetical
    {key|t}    -> "Rajab and Younis (2021)"        textual, author is the subject
    {a;b}      -> "(Mahmud et al. 2022; Gupta et al. 2017)"

check_paper.py uses REFERENCES + the placeholders to prove that every citation
resolves and every reference is cited.
"""

import re

# --------------------------------------------------------------------------- #
# References -- Chicago, in the form the guide asked for:
#   Author, Full First M., and Co-Author. "Sentence-case title."
#   Journal Vol, no. N (Year): pages.
# Given names verified against the PDFs in ../Papers for BBN and
# ../Papers for Fog Computing.  Where a paper is published with initials only
# (Aguilera, Reyana) the published form is kept rather than invented.
# --------------------------------------------------------------------------- #

REFERENCES = {
    "aguilera2011": dict(
        short="Aguilera et al.", year=2011, sort=("Aguilera", "P. A."),
        text='Aguilera, P. A., A. Fernández, R. Fernández, R. Rumí, and A. Salmerón. '
             '"Bayesian networks in environmental modelling." '
             '•Environmental Modelling & Software• 26, no. 12 (2011): 1376-1388.'),
    "fenton2007": dict(
        short="Fenton, Neil, and Marquez", year=2007, sort=("Fenton", "Norman"),
        text='Fenton, Norman, Martin Neil, and David Marquez. '
             '"Using Bayesian networks to predict software defects and reliability." In '
             '•Proceedings of the 5th International Conference on Mathematical Methods in '
             'Reliability (MMR07)•. Glasgow, UK, 2007.'),
    "gupta2017": dict(
        short="Gupta et al.", year=2017, sort=("Gupta", "Harshit"),
        text='Gupta, Harshit, Amir Vahid Dastjerdi, Soumya K. Ghosh, and Rajkumar Buyya. '
             '"iFogSim: A toolkit for modeling and simulation of resource management techniques '
             'in the Internet of Things, Edge and Fog computing environments." '
             '•Software: Practice and Experience• 47, no. 9 (2017): 1275-1296.'),
    "gupta2025": dict(
        short="Gupta and Singh", year=2025, sort=("Gupta", "Shally"),
        text='Gupta, Shally, and Nanhay Singh. '
             '"Optimized fault-tolerant placement of microservices in distributed fog networks." '
             '•International Journal of Data Informatics and Intelligent Computing• 4, '
             'no. 4 (2025): 52-69.'),
    "hasan2024": dict(
        short="Hasan and Zeebaree", year=2024, sort=("Hasan", "Dathar Abas"),
        text='Hasan, Dathar Abas, and Subhi R. M. Zeebaree. '
             '"Proactive fault tolerance in distributed cloud systems: A review of predictive '
             'and preventive techniques." '
             '•Indonesian Journal of Computer Science• 13, no. 2 (2024).'),
    "mahmud2022": dict(
        short="Mahmud et al.", year=2022, sort=("Mahmud", "Redowan"),
        text='Mahmud, Redowan, Samodha Pallewatta, Mohammad Goudarzi, and Rajkumar Buyya. '
             '"iFogSim2: An extended iFogSim simulator for mobility, clustering, and microservice '
             'management in edge and fog computing environments." '
             '•Journal of Systems and Software• 190 (2022): 111351.'),
    "naim2016": dict(
        short="Naim, Zain, and Jalil", year=2016, sort=("Naim", "Mohd Hariz"),
        text='Naim, Mohd Hariz, Jasni Mohamad Zain, and Kamarularifin Abd Jalil. '
             '"Fault tolerance mechanism for software application through fog computing as '
             'middleware." •International Journal of Computing and Digital Systems• 5, '
             'no. 3 (2016).'),
    "pang2025": dict(
        short="Pang and Abramov", year=2025, sort=("Pang", "Bo"),
        text='Pang, Bo, and Evgeny S. Abramov. '
             '"Reliability analysis and parameter selection for IoT communication based on deep '
             'learning." •Eng• 6, no. 8 (2025): 171.'),
    "pearl1988": dict(
        short="Pearl", year=1988, sort=("Pearl", "Judea"),
        text='Pearl, Judea. •Probabilistic Reasoning in Intelligent Systems: Networks of '
             'Plausible Inference•. San Mateo, CA: Morgan Kaufmann, 1988.'),
    "rajab2021": dict(
        short="Rajab and Younis", year=2021, sort=("Rajab", "Hadeel T."),
        text='Rajab, Hadeel T., and Manal F. Younis. '
             '"Dynamic fault tolerance aware scheduling for healthcare system on fog computing." '
             '•Iraqi Journal of Science• (2021): 308-318.'),
    "reyana2023": dict(
        short="Reyana et al.", year=2023, sort=("Reyana", "A."),
        text='Reyana, A., Sandeep Kautish, Khalid Abdulaziz Alnowibet, Hossam M. Zawbaa, and '
             'Ali Wagdy Mohamed. "Opportunities of IoT in fog computing for high fault tolerance '
             'and sustainable energy optimization." '
             '•Sustainability• 15, no. 11 (2023): 8702.'),
    "syed2022": dict(
        short="Syed et al.", year=2022, sort=("Syed", "Sidra Abid"),
        text='Syed, Sidra Abid, Munaf Rashid, Samreen Hussain, Fahad Azim, Hira Zahid, Asif Umer, '
             'Abdul Waheed, Mahdi Zareei, and Cesar Vargas-Rosales. '
             '"QoS aware and fault tolerance based software-defined vehicular networks using '
             'cloud-fog computing." •Sensors• 22, no. 1 (2022): 401.'),
    "weber2012": dict(
        short="Weber et al.", year=2012, sort=("Weber", "Philippe"),
        text='Weber, Philippe, Gabriela Medina-Oliva, Christophe Simon, and Benoît Iung. '
             '"Overview on Bayesian networks applications for dependability, risk analysis and '
             'maintenance areas." '
             '•Engineering Applications of Artificial Intelligence• 25, no. 4 (2012): '
             '671-682.'),
}

_PLACEHOLDER = re.compile(r"\{([a-z]+[0-9]{4}(?:;[a-z]+[0-9]{4})*(?:\|t)?)\}")


def cite(s):
    """Expand {key}, {key|t} and {a;b} placeholders into Chicago author-date form."""
    def sub(m):
        body = m.group(1)
        textual = body.endswith("|t")
        if textual:
            body = body[:-2]
        keys = body.split(";")
        for k in keys:
            if k not in REFERENCES:
                raise KeyError("unknown citation key: %r" % k)
        if textual:
            if len(keys) != 1:
                raise ValueError("textual citation takes one key: %r" % body)
            r = REFERENCES[keys[0]]
            return "%s (%d)" % (r["short"], r["year"])
        return "(" + "; ".join(
            "%s %d" % (REFERENCES[k]["short"], REFERENCES[k]["year"]) for k in keys) + ")"
    return _PLACEHOLDER.sub(sub, s)


def used_keys(s):
    out = set()
    for m in _PLACEHOLDER.finditer(s):
        body = m.group(1)
        if body.endswith("|t"):
            body = body[:-2]
        out.update(body.split(";"))
    return out


def references_sorted():
    return [REFERENCES[k]["text"]
            for k in sorted(REFERENCES, key=lambda k: REFERENCES[k]["sort"])]


# --------------------------------------------------------------------------- #
# Front matter
# --------------------------------------------------------------------------- #

TITLE = ("Bayesian Belief Network-Driven Proactive Fault Tolerance for "
         "Microservice Placement in Fog Computing Environments")

AUTHORS = [
    ("Rahul Akshath Rajendran", "3122235002095"),
    ("Mohit R", "3122235002072"),
    ("Dr Sabireen H, Assistant Professor", "Project Guide"),
]

AFFILIATION = [
    "Department of Information Technology",
    "SSN College of Engineering, Kalavakkam, Chennai, India",
    "{rahul2310405, mohit2310893, sabireenh}@ssn.edu.in",
]

ABSTRACT = (
    "Fog computing extends cloud services to the network edge in order to satisfy the "
    "stringent latency requirements of Internet-of-Things (IoT) applications such as remote "
    "cardiac monitoring, intelligent video surveillance and immersive gaming. The fog nodes "
    "that host these applications are, however, heterogeneous, resource-constrained and "
    "considerably less reliable than data-centre hardware, so the failure of a single node can "
    "breach the deadline of a safety-critical service. Existing fog fault-tolerance mechanisms "
    "are predominantly reactive: checkpoint-and-restore, primary-backup replication, "
    "acknowledgement-based retransmission and heartbeat-driven failover all take effect only "
    "after a fault has already manifested, and therefore incur a recovery latency that "
    "latency-critical workloads cannot absorb. The few proactive schemes reported in the "
    "literature trigger migration on a single fixed threshold applied to a single metric, which "
    "neither captures the multi-causal nature of fog node degradation nor tolerates missing or "
    "noisy telemetry. This paper proposes a Bayesian Belief Network (BBN) driven proactive "
    "fault-tolerance framework for microservice placement in fog environments. The framework "
    "contributes: (i) a causal BBN that infers the posterior probability of node failure within "
    "a look-ahead horizon from partial and noisy telemetry and exposes an explainable causal "
    "chain rather than an opaque score; (ii) a risk-aware placement and expected-loss migration "
    "policy that replaces fixed thresholds with a cost-sensitive decision rule; and (iii) a "
    "criticality-aware selective replication scheme that allocates standby capacity in "
    "proportion to predicted risk and application criticality. The framework is being realised "
    "as an extension to the iFogSim2 simulator and will be evaluated on the Melbourne CBD "
    "edge-resource and user-mobility datasets against representative reactive and "
    "threshold-based baselines.")

INDEX_TERMS = (
    "Bayesian belief network, fog computing, proactive fault tolerance, microservice "
    "placement, reliability, iFogSim2, Internet of Things, probabilistic inference.")


# --------------------------------------------------------------------------- #
# Body.  Block kinds:
#   ("h1", numeral, title)          ("h2", letter, title)
#   ("p", text)                     ("bullet", text)        ("num", text)
#   ("eq", expr, "(1)")             -- expr uses _[...] for subscripts
#   ("where", [(symbol, definition), ...])
#   ("fig", filename, width_in, height_in)
#   ("cap", "Fig. 1.", text)
#   ("tcap", "TABLE I.", TITLE)     ("table", rows)
# --------------------------------------------------------------------------- #

BODY = [

# ======================= I. INTRODUCTION =======================
("h1", "I.", "Introduction"),

("p",
 "The proliferation of latency-sensitive Internet-of-Things (IoT) applications has exposed a "
 "structural limitation of the centralised cloud model. An electrocardiogram stream that must "
 "be classified within a few tens of milliseconds, or a surveillance feed that must trigger a "
 "pan-tilt-zoom actuator before a tracked object leaves the frame, cannot tolerate the "
 "round-trip delay to a distant data centre. Fog computing addresses this by interposing a "
 "tier of computational resources — gateways, routers, base stations and micro data centres — "
 "between the end devices and the cloud, so that application modules execute within one or two "
 "network hops of the sensors that feed them {mahmud2022;gupta2017}."),

("p",
 "This proximity is purchased at the cost of reliability. Whereas a cloud data centre offers "
 "redundant power, cooling and network paths, a fog node is frequently a commodity device "
 "deployed in an uncontrolled environment, sharing a constrained uplink, operating on limited "
 "or battery power, and serving mobile users whose handovers continually reshape the load it "
 "carries. Empirical studies of distributed infrastructure consistently report that such nodes "
 "fail far more often than data-centre hosts, and that failures are rarely instantaneous: a "
 "node typically degrades through a period of rising utilisation, thermal stress, queue "
 "build-up and link deterioration before it becomes unresponsive {hasan2024}. That degradation "
 "window is precisely the opportunity that a proactive mechanism should exploit."),

("p",
 "The fog fault-tolerance literature has, however, largely declined to exploit it. The dominant "
 "mechanisms — dynamic checkpointing {rajab2021}, acknowledgement-based retransmission "
 "{syed2022}, heartbeat monitoring with failover to a replicated backup {naim2016} — are "
 "reactive by construction. They detect the absence of a response or the loss of a message, and "
 "then initiate recovery. For a best-effort workload this is adequate; for a cardiac arrhythmia "
 "classifier it means that the very interval during which an alarm should have been raised is "
 "consumed by state restoration. Even the schemes that describe themselves as proactive reduce "
 "the decision to a single scalar comparison, such as creating a checkpoint whenever CPU "
 "utilisation exceeds twenty per cent {rajab2021}, or migrating a microservice when a composite "
 "health index falls below a preset fault threshold {gupta2025}. A fixed threshold on a single "
 "metric is simultaneously too eager and too late: it fires on transient load spikes that "
 "resolve without incident, and it stays silent when a node is degrading along a dimension the "
 "metric does not observe."),

("p",
 "What is missing is a principled representation of uncertainty. Fog node health is a latent "
 "condition, inferred indirectly from several noisy and mutually dependent indicators, some of "
 "which may be unavailable at any given instant because a telemetry agent has itself been "
 "affected. Bayesian Belief Networks (BBNs) are the canonical formalism for exactly this "
 "situation. A BBN encodes causal structure as a directed acyclic graph and quantifies it with "
 "conditional probability tables, supports exact inference of a posterior over an unobserved "
 "variable given whatever evidence happens to be available, degrades gracefully when evidence "
 "is partial by marginalising over the missing variables, and permits expert knowledge to be "
 "combined with observed data {pearl1988}. BBNs have been applied with documented success to "
 "software defect and reliability prediction {fenton2007}, to dependability, risk analysis and "
 "maintenance in industrial systems {weber2012}, and to environmental modelling under sparse "
 "observation {aguilera2011}. Their transfer to fog reliability management is natural and, to "
 "the best of our knowledge, unexplored."),

("p",
 "This paper proposes a proactive fault-tolerance framework in which a BBN estimates, for every "
 "fog node, the probability that it will fail within a look-ahead horizon, and in which that "
 "posterior drives placement, migration and redundancy decisions. The contributions are "
 "threefold:"),

("bullet",
 "A BBN-based failure-prediction engine that infers the posterior probability of imminent node "
 "failure from heterogeneous telemetry, tolerates missing and noisy evidence, exposes the "
 "causal chain responsible for each prediction, and adapts its conditional probability tables "
 "online from observed outcomes."),
("bullet",
 "A risk-aware placement and expected-loss migration policy in which the BBN posterior enters a "
 "multi-objective utility over latency, energy and risk, and in which migration is triggered by "
 "a cost-sensitive expected-loss comparison rather than a fixed threshold."),
("bullet",
 "A criticality-aware selective replication scheme that allocates standby capacity in "
 "proportion to the product of predicted risk and application criticality, so that redundancy "
 "is spent where it changes the outcome instead of uniformly across all modules."),

("p",
 "The remainder of the paper is organised as follows. Section II surveys reactive and proactive "
 "fault tolerance in fog computing and the use of BBNs in reliability engineering, and "
 "identifies the research gap. Section III presents the proposed framework, its block diagram "
 "and its workflow. Section IV describes the simulation environment, the datasets and the "
 "evaluation plan. Section V concludes and outlines the work planned for Phase II."),

# ======================= II. LITERATURE SURVEY =======================
("h1", "II.", "Literature Survey"),
("h2", "A.", "Reactive Fault Tolerance in Fog Environments"),

("p",
 "{rajab2021|t} propose a dynamic fault-tolerance-aware scheduling approach for healthcare "
 "systems on fog computing. Their scheme combines Dynamic Weighted Round Robin (DWRR) "
 "scheduling, in which each fog server is assigned a weight derived from its current CPU and "
 "memory utilisation, with primary-backup replication and dynamic checkpointing. Rather than "
 "checkpointing at fixed intervals, the system creates a checkpoint whenever CPU utilisation "
 "crosses a threshold of 0.2, and every checkpoint is replicated to a designated backup server "
 "so that execution can resume from the most recent saved state if the primary fails. The work "
 "demonstrates a clear throughput benefit from load-aware distribution and a reduced "
 "re-execution cost from adaptive checkpointing. Its limitations are equally clear: the "
 "threshold is fixed and workload-independent, the backup server consumes storage whether or "
 "not a failure ever occurs, evaluation is confined to a healthcare scenario, and the system "
 "possesses no capacity to anticipate a failure before the threshold condition is met."),

("p",
 "{syed2022|t} address reliability in vehicular networks through QAFT-SDVN, which integrates "
 "software-defined networking with a cloud-fog substrate. A centralised SDN controller "
 "classifies each message by type, deadline and size, assigns a priority accordingly, and "
 "routes safety-critical traffic such as ambulance alerts to the nearest fog node while "
 "queueing non-safety traffic at lower priority. Fault tolerance is provided by "
 "acknowledgement-based retransmission: the sender awaits an ACK and retransmits if none "
 "arrives within the expected interval. The approach roughly halves response time for "
 "prioritised traffic and reduces the message failure ratio, but recovery is purely reactive "
 "and consists solely of retransmission. No backup execution, checkpointing or replica "
 "management is provided, the additional ACK traffic inflates network load, and the design is "
 "specialised to the vehicular domain."),

("p",
 "{naim2016|t} place a fog middleware between users and application servers to improve "
 "availability. The middleware polls the primary server with heartbeat requests, maintains a "
 "replicated backup containing the same application and data, and on detecting the absence of a "
 "heartbeat response activates the backup and transparently redirects all subsequent user "
 "requests to it. A prototype recovered from three of four simulated failure scenarios without "
 "manual intervention. The mechanism is nevertheless strictly detection-driven: it waits until "
 "the server has already stopped responding, considers only a single backup so that a "
 "correlated double failure is unrecoverable, imposes continuous heartbeat overhead, and "
 "performs no load-aware server selection."),

("h2", "B.", "Optimisation-Based and Threshold-Based Proactive Placement"),

("p",
 "{reyana2023|t} propose QoS-EO, a hybrid task-placement approach that combines Particle Swarm "
 "Optimisation with the Artificial Bee Colony algorithm. Candidate solutions encode "
 "task-to-node assignments and are scored by a fitness function over response time, energy "
 "consumption, reliability, throughput and execution cost; PSO drives the swarm towards better "
 "assignments while ABC's worker, onlooker and scout phases refine and diversify the search. "
 "Better placement reduces node overload and thereby improves fault tolerance indirectly. The "
 "cost is computational: running two metaheuristics in sequence is expensive, performance is "
 "sensitive to inertia weight and learning-factor settings, and the search space grows "
 "unfavourably as the number of IoT devices increases. Crucially, reliability enters only as a "
 "static term in a fitness function; the scheme does not estimate the probability that any "
 "particular node will fail."),

("p",
 "{gupta2025|t} come closest to a proactive design. Their Fault-Tolerance Aware Placement "
 "Algorithm (FTAPA) selects a primary fog node using a Dynamic Connections Load Balancer that "
 "evaluates active connections, resource availability and node health; it simultaneously "
 "pre-selects a backup node, monitors the health of the primary against a computed fault "
 "threshold, and migrates the microservice to the backup before the primary fails outright. "
 "This anticipatory migration measurably reduces service downtime relative to detection-driven "
 "recovery. The authors themselves identify the residual weaknesses: backup nodes consume "
 "memory and processing capacity even when failures never materialise, continuous health "
 "monitoring adds overhead, and the fault threshold is difficult to set — too low and migration "
 "happens too late, too high and the system migrates unnecessarily. The threshold is a point "
 "estimate of a quantity that is inherently uncertain, and no confidence is attached to it."),

("h2", "C.", "Bayesian Belief Networks for Reliability Prediction"),

("p",
 "A Bayesian Belief Network is a directed acyclic graph whose vertices are random variables and "
 "whose edges encode direct probabilistic dependence, quantified by a conditional probability "
 "table for each vertex given its parents. The joint distribution factorises as the product of "
 "these local conditionals, which makes exact posterior inference tractable for the sparse "
 "structures that arise when causal knowledge is available. The foundational treatment by "
 "{pearl1988|t} established both the semantics of such networks and the message-passing "
 "algorithms that make inference practical."),

("p",
 "The formalism has been applied extensively to reliability. {fenton2007|t} use BNs to predict "
 "software defects and reliability, arguing that the ability to combine causal process factors "
 "with quantitative measures, and qualitative expert judgement with observed data, overcomes "
 "well-known limitations of classical metrics-based models; the approach has been deployed by "
 "Motorola, Siemens and Philips with reported accuracy. {weber2012|t} survey a decade of BN "
 "applications to dependability, risk analysis and maintenance, and document a sustained growth "
 "in adoption attributable to advantages over Markov chains, fault trees and Petri nets — "
 "specifically the capacity to model complex systems with heterogeneous variables, to perform "
 "both predictive and diagnostic reasoning, and to update beliefs incrementally as evidence "
 "arrives. {aguilera2011|t} review BNs in environmental modelling, a domain characterised, like "
 "fog telemetry, by sparse and unreliable observation, and emphasise the graceful handling of "
 "missing data. More recently, {pang2025|t} analyse reliability and parameter selection for IoT "
 "communication using deep learning, confirming the importance of predictive reliability "
 "modelling for IoT while also illustrating the interpretability cost of black-box predictors — "
 "a cost that a BBN does not impose."),

("p",
 "Two properties make BBNs particularly appropriate for fog node health. First, inference is "
 "well-defined under partial evidence: unobserved variables are marginalised out, so a "
 "prediction is still produced when a telemetry channel is unavailable, which is precisely the "
 "condition likely to coincide with impending failure. Second, the network is diagnostic as "
 "well as predictive: having inferred a high failure probability, the same model identifies "
 "which parent variables are most responsible, yielding an explanation that an operator can act "
 "on and that a threshold comparison cannot provide."),

("h2", "D.", "Comparative Analysis"),
("tcap", "TABLE I.", "COMPARATIVE ANALYSIS OF FAULT-TOLERANCE MECHANISMS IN FOG COMPUTING"),
("table", [
    ["Study", "Technique", "Nature", "Key Limitation"],
    ["{rajab2021|t}",
     "DWRR scheduling + dynamic checkpointing + primary-backup replication",
     "Reactive",
     "Fixed CPU threshold (0.2); backup storage always reserved; no failure prediction"],
    ["{syed2022|t}",
     "SDN priority scheduling + ACK-based retransmission",
     "Reactive",
     "Recovery limited to retransmission; extra ACK traffic; vehicular domain only"],
    ["{naim2016|t}",
     "Fog middleware + heartbeat monitoring + replicated failover",
     "Reactive",
     "Acts only after server stops responding; single backup; continuous heartbeat overhead"],
    ["{reyana2023|t}",
     "QoS-EO hybrid PSO + ABC task placement",
     "Preventive (static)",
     "High computational cost; parameter-sensitive; reliability is a static fitness term"],
    ["{gupta2025|t}",
     "FTAPA: DCLB node selection + threshold-based proactive migration",
     "Proactive (threshold)",
     "Fault threshold hard to tune; backup reserved per microservice; monitoring overhead"],
    ["This work",
     "Causal BBN inference + expected-loss migration + criticality-aware replication",
     "Proactive (probabilistic)",
     "Requires CPT elicitation and online calibration; addressed by literature priors and "
     "Bayesian parameter learning"],
]),

("h2", "E.", "Inferences Drawn from the Survey"),
("p",
 "Read together rather than individually, the five systems support six inferences that shape "
 "the design proposed in this paper."),

("num",
 "Fault tolerance in fog computing is overwhelmingly reactive. Three of the five systems act "
 "only once a fault has been detected — by a missing heartbeat {naim2016}, an unacknowledged "
 "message {syed2022} or a threshold breach during execution {rajab2021} — so the recovery "
 "latency is incurred at precisely the moment the application can least absorb it."),
("num",
 "Acting before failure demonstrably helps. FTAPA {gupta2025} reports measurably reduced "
 "service downtime from migrating a microservice ahead of a predicted failure. The principle is "
 "sound, and this paper builds on it rather than disputing it."),
("num",
 "Where prediction is attempted it is deterministic and single-metric. Both {rajab2021|t} and "
 "{gupta2025|t} reduce the decision to a comparison of one scalar against a preset threshold, "
 "and the authors of both report the difficulty of choosing that threshold as an open "
 "limitation."),
("num",
 "Node health is treated as directly observable. No surveyed system represents it as a latent "
 "condition that must be inferred from indicators which are noisy, mutually dependent and "
 "sometimes absent — despite the fact that a telemetry channel is most likely to fall silent "
 "exactly when the node hosting it is degrading."),
("num",
 "Redundancy is provisioned uniformly rather than by risk. A backup is reserved per task "
 "{rajab2021}, per server {naim2016} or per microservice {gupta2025}, consuming standby "
 "capacity irrespective of whether a failure is likely; {gupta2025|t} identify this as a "
 "standing cost of their own design."),
("num",
 "Reliability enters optimisation only as a static term. QoS-EO {reyana2023} scores reliability "
 "within a fitness function but never estimates the probability that a particular node will "
 "fail within a particular interval, so its reliability term cannot respond to changing "
 "conditions."),

("p",
 "Taken together, these inferences point to a single conclusion: what the field lacks is a "
 "probabilistic and explainable estimate of imminent node failure, together with a decision "
 "rule that spends migration and redundancy in proportion to that estimate."),

("h2", "F.", "Selection of the Modelling Approach"),
("p",
 "Several established formalisms could in principle supply such an estimate, and the choice "
 "among them was made against four requirements imposed by the problem: the method must "
 "function without a large corpus of labelled failures, since none exists for fog topologies; "
 "it must continue to produce an estimate when part of the telemetry is unavailable; it must "
 "emit a calibrated probability, because the decision rule proposed in Section III multiplies "
 "that probability by a cost; and it must be interpretable, so that an operator can act on the "
 "reason for a prediction rather than only its magnitude. Table II summarises the candidates "
 "against these requirements."),

("tcap", "TABLE II.", "SELECTION OF THE MODELLING APPROACH"),
("table", [
    ["Candidate approach", "Meets requirements?", "Principal reason"],
    ["Fixed threshold rules", "No",
     "Single metric; no uncertainty; silent under missing telemetry; threshold hard to tune"],
    ["Metaheuristics (PSO, ABC, GA)", "No",
     "Optimises assignment but never estimates node failure; costly and parameter-sensitive"],
    ["Markov chains / fault trees", "Partly",
     "State-space explosion; poor with partial evidence; static structure"],
    ["ML classifiers / deep learning", "Partly",
     "Needs a large labelled failure corpus; miscalibrated outputs; requires imputation; opaque"],
    ["Bayesian Belief Network", "Yes",
     "Priors usable without data; marginalises missing evidence; calibrated posterior; causally "
     "explainable"],
]),

("p",
 "Fixed threshold rules fail the first three requirements outright. Metaheuristic placement "
 "optimises an assignment but never estimates failure at all. Markov chains and fault trees, "
 "the classical instruments of reliability analysis, suffer state-space explosion on systems of "
 "this size and accommodate partial evidence poorly — precisely the comparison {weber2012|t} "
 "draw in concluding that Bayesian networks have displaced them across dependability and "
 "maintenance applications. Discriminative machine learning, including the deep-learning "
 "approach that {pang2025|t} apply to IoT reliability, achieves high accuracy where abundant "
 "labelled data exists, but requires an input value in every feature slot and so must impute "
 "the very channels whose absence is diagnostic; its confidence scores are also well known to "
 "be poorly calibrated, which matters when the output is to be multiplied by a cost."),

("p",
 "A Bayesian Belief Network satisfies all four requirements simultaneously. Its conditional "
 "probability tables can be initialised from expert judgement and published reliability priors "
 "and then refined from observation, so it is usable from the first epoch. Inference "
 "marginalises over unobserved variables, so partial telemetry degrades the sharpness of the "
 "posterior without preventing a decision. The output is a probability in the formal sense, "
 "suitable for composition with a loss function. And because the graph encodes causal "
 "structure, the same inference that produces the posterior also identifies the parent "
 "configuration most responsible for it. This combination, rather than any single property, is "
 "why the Bayesian Belief Network was selected."),

("h2", "G.", "Research Gap"),
("p",
 "Table I makes the gap explicit. Every surveyed fault-tolerance mechanism either reacts to a "
 "fault that has already occurred, or anticipates it using a deterministic comparison against a "
 "fixed threshold on a small number of directly observed metrics. None represents node health "
 "as a latent variable, none quantifies the uncertainty of its own prediction, none continues "
 "to reason when telemetry is incomplete, and none uses a predicted failure probability to "
 "modulate how much redundancy a given module deserves. Conversely, the BBN literature provides "
 "a mature apparatus for probabilistic reliability inference under uncertainty, but has not "
 "been applied to the fog placement problem. This paper addresses that intersection."),
]

BODY += [

# ======================= III. PROPOSED METHODOLOGY =======================
("h1", "III.", "Proposed Methodology"),
("h2", "A.", "System Architecture"),

("p",
 "Fig. 1 presents the proposed framework. It preserves the conventional three-tier fog topology "
 "— an IoT and end-device layer, a fog layer and a cloud layer — and introduces a Proactive "
 "Fault-Tolerance Controller that is co-located with the Fog Orchestration Node at the proxy "
 "tier of the hierarchy, where it has visibility of an entire cluster of fog nodes without "
 "incurring the round-trip delay of a cloud-hosted controller."),

("fig", "figures/paper_fig1_architecture.png", 3.60, 5.43),
("cap", "Fig. 1.",
 "Block diagram of the proposed BBN-driven proactive fault-tolerance framework. Telemetry from "
 "the fog layer is discretised and supplied as evidence to the BBN inference engine (N1); the "
 "resulting posterior drives risk-aware placement and expected-loss migration (N2) and "
 "criticality-aware selective replication (N3). Observed outcomes are fed back for online "
 "conditional-probability-table adaptation."),

("p",
 "At the lowest layer, sensors and end devices — electrocardiogram and electroencephalogram "
 "sensors, surveillance cameras and mobile user equipment — emit tuples that are consumed by "
 "application modules placed on fog nodes. Every fog node hosts a lightweight telemetry agent "
 "that periodically reports the observable indicators of node condition. The controller "
 "consumes this telemetry through the four stages drawn inside it in Fig. 1. A pre-processing "
 "and discretisation stage cleans, aligns and bins the continuous measurements into the ordinal "
 "states over which the network is defined. The BBN inference engine (N1) then computes, for "
 "each node, the posterior probability of failure within the look-ahead horizon. The risk-aware "
 "placement and migration module (N2) converts those posteriors into placement and migration "
 "decisions, and the criticality-aware replication module (N3) decides how much standby "
 "capacity each placed module should be granted. Decisions are actuated back onto the fog nodes "
 "as module placements, live migrations and replica allocations, while modules that no fog node "
 "can safely host are offloaded to the cloud. Observed outcomes — whether a node predicted to "
 "fail actually failed — are fed back to update the conditional probability tables, and the "
 "updated tables are synchronised with a cloud-hosted repository so that knowledge learned in "
 "one cluster benefits others."),

("h2", "B.", "Novelty 1: BBN-Based Failure Prediction Engine"),

("p",
 "The core of the framework is a Bayesian Belief Network over fog-node condition, whose "
 "structure is shown in Fig. 2. The network is organised in three tiers. Observable evidence "
 "variables correspond directly to telemetry: CPU utilisation, RAM utilisation, tuple queue "
 "depth, uplink latency, jitter and packet loss, residual energy or battery level, ambient or "
 "die temperature, and the handover rate induced by user mobility. Intermediate latent "
 "variables aggregate these into interpretable conditions — computational stress, network "
 "degradation, energy stress and thermal stress. A single target variable, the failure of the "
 "node within the horizon, is the child of the intermediate tier, additionally conditioned on a "
 "node-reliability prior derived from the device class and its historical mean time between "
 "failures."),

("p",
 "The edges follow the physical origin of each indicator. CPU utilisation, RAM utilisation and "
 "tuple queue depth are the parents of computational stress; uplink latency, jitter with packet "
 "loss, and handover rate are the parents of network degradation; residual energy is the parent "
 "of energy stress; and temperature is the parent of thermal stress. Those four latent "
 "variables, together with the node-reliability prior, are in turn the parents of the target. "
 "Each variable is discretised into a small ordinal state space, { low, medium, high }, so that "
 "conditional probability tables remain compact and interpretable. Given the set of evidence "
 "actually observed at a decision epoch, the engine computes the posterior"),

("eq", "P(F = fail | e)  =  [ Σ_[u] Π_[V] P(V | pa(V)) ]  /  P(e)", "(1)"),
("where", [
    ("P(F = fail | e)",
     "the posterior probability, given the evidence actually observed, that the node fails "
     "within the look-ahead horizon; this single number is what drives every decision in "
     "Sections III.C and III.D."),
    ("F",
     "the target variable of the network, taking the states fail and not-fail, where fail means "
     "\u201cthe node becomes unable to host its modules at some point within the next \u0394t\u201d."),
    ("\u0394t",
     "the look-ahead horizon: the length of the interval over which failure is predicted, "
     "chosen so that it exceeds the time needed to migrate a module."),
    ("e",
     "the evidence, that is, the set of observed states of those telemetry variables that "
     "actually reported at this decision epoch. Channels that failed to report are simply "
     "absent from e rather than imputed."),
    ("\u03a3_[u]",
     "the summation over u, and the step that gives the network its tolerance of missing "
     "telemetry: every possible combination of values for the variables that were not observed "
     "is enumerated and its contribution added up, which is what marginalising them out means."),
    ("u",
     "one such joint assignment of states to the unobserved variables \u2014 the latent "
     "conditions, plus any telemetry channel missing from e."),
    ("\u03a0_[V]",
     "the product over V, taken across every variable of the network. Multiplying the local "
     "conditional probabilities together in this way reconstitutes the full joint distribution, "
     "and is what makes exact inference tractable on a sparse graph."),
    ("V",
     "an individual variable of the network. The product runs over all fourteen of them: the "
     "eight evidence variables, the four latent condition variables, the node-reliability prior "
     "and F itself."),
    ("pa(V)",
     "the parents of V in the directed acyclic graph, so that P(V | pa(V)) is exactly the "
     "conditional probability table stored at V."),
    ("P(e)",
     "the normalising constant, obtained by summing the same expression over both states of F "
     "as well as over u, so that the two posterior probabilities sum to one."),
]),

("p",
 "This is the property that distinguishes the approach from a threshold rule and from a "
 "discriminative classifier: a missing telemetry channel widens the sum over u and so degrades "
 "the sharpness of the posterior, but it does not prevent a decision from being made, whereas a "
 "threshold on an unreported metric simply never fires."),

("p",
 "With one exception, every evidence variable is derived from a quantity the simulator "
 "described in Section IV already maintains. CPU utilisation is allocated MIPS against host capacity per scheduling "
 "interval; RAM utilisation comes from the memory provisioner; queue depth is the pending tuple "
 "queue on the device; uplink latency and jitter combine the configured link delay with "
 "queueing delay; residual energy is accumulated consumption against a configured budget; and "
 "handover rate is the count of parent reassignments per window, while the node-reliability "
 "prior is fixed by device class and configured mean time between failures. Temperature is read "
 "wherever a node exposes a thermal sensor; where none is exposed, as on the simulated devices "
 "of Section IV, that variable is left unobserved and is marginalised out by (1) — the "
 "partial-evidence case the network is built to absorb, rather than a gap that has to be filled "
 "in. Continuous quantities are discretised using quantiles of the baseline runs, so that a "
 "state such as high CPU utilisation is defined relative to the observed distribution of this "
 "topology rather than by an arbitrary cut-off."),

("p",
 "Conditional probability tables are initialised from the reliability literature "
 "{fenton2007;weber2012;pang2025} and from expert elicitation over the device classes present "
 "in the topology, since no public failure trace exists for the fog topology under study. They "
 "are subsequently refined online: each prediction is paired with the outcome eventually "
 "observed, and the relevant table entries are updated by Bayesian parameter learning with "
 "Dirichlet priors, so the model becomes progressively better calibrated to the deployment it "
 "governs. Because the network is causal rather than correlational, the posterior is accompanied "
 "by the most influential parent configuration, giving an explanation of the form “failure "
 "probability is high principally because network degradation is high given elevated jitter and "
 "packet loss” — information that supports both operator trust and the choice of remedy."),

("fig", "figures/paper_fig2_bbn.png", 5.60, 4.31),
("cap", "Fig. 2.",
 "Structure of the Bayesian Belief Network over fog-node condition. Observable telemetry forms "
 "the evidence tier; four latent variables aggregate it into interpretable conditions; the "
 "target is failure of the node within the horizon Δt, additionally conditioned on a "
 "node-reliability prior derived from the device class and its historical mean time between "
 "failures. Evidence that is unavailable at a decision epoch is marginalised out rather than "
 "imputed."),

("h2", "C.", "Novelty 2: Risk-Aware Placement and Expected-Loss Migration"),

("p",
 "The posterior produced by the BBN is not consumed as a binary alarm. For placement, the "
 "controller scores each candidate node for a given module by a normalised multi-objective "
 "utility"),

("eq",
 "U(m, n)  =  α · (1 − L̂(m, n))  +  β · (1 − Ê(m, n))  +  γ · (1 − P(F_[n] = fail | e))",
 "(2)"),
("where", [
    ("U(m, n)",
     "the placement utility of hosting module m on node n, on a scale of 0 to 1; the controller "
     "places each module on the feasible node of highest utility."),
    ("m",
     "the application module, that is, the microservice awaiting placement."),
    ("n",
     "the candidate fog node being scored."),
    ("L̂(m, n)",
     "the predicted end-to-end latency of running m on n, normalised to the interval 0 to 1 "
     "across the candidate nodes; it is subtracted from 1 so that low latency raises the "
     "utility."),
    ("Ê(m, n)",
     "the predicted energy cost of running m on n, normalised the same way and likewise "
     "subtracted from 1."),
    ("P(F_[n] = fail | e)",
     "the posterior failure probability of node n from (1); the term 1 − P(F_[n] = fail | e) is "
     "therefore the reliability of n as the network currently estimates it."),
    ("α, β, γ",
     "non-negative weights on latency, energy and risk respectively, summing to 1 and "
     "configured per application class."),
]),

("p",
 "The weights are what make the utility application-aware. A latency-critical health module is "
 "configured with a large γ, so it is steered away from nodes the network considers fragile "
 "even when they are computationally attractive; a best-effort module weights latency and "
 "energy more heavily instead."),

("p",
 "For migration, the framework replaces the fixed fault threshold of prior work with a "
 "cost-sensitive decision. Migrating a running module is not free: it consumes bandwidth, "
 "incurs a brief service interruption and may place the module on a less favourable node. "
 "Migration is therefore triggered only when the expected loss from remaining exceeds the cost "
 "of moving,"),

("eq",
 "P(F_[n] = fail | e) · C_[fail](m)  >  C_[mig](m, n → n′)",
 "(3)"),
("where", [
    ("P(F_[n] = fail | e) · C_[fail](m)",
     "the left-hand side: the expected loss of leaving m on n for one more decision epoch, "
     "being the probability of the loss multiplied by its magnitude."),
    ("C_[fail](m)",
     "the penalty attributable to losing module m — itself a function of the application's "
     "criticality and deadline, so that a cardiac classifier carries a far larger value than a "
     "best-effort module."),
    ("C_[mig](m, n → n′)",
     "the estimated cost of relocating m from its current node n to the alternative node n′, "
     "covering the bandwidth needed to transfer state, the brief interruption to service, and "
     "any utility given up by moving to a less favourable node."),
    ("n′",
     "the best alternative node, namely the node that maximises the utility U(m, n) of (2) "
     "among the feasible candidates."),
]),

("p",
 "The decision rule is self-calibrating: a module whose failure is cheap tolerates a high "
 "failure probability before it is moved, while a module whose failure is expensive is moved on "
 "comparatively weak evidence. Rearranging (3) makes this explicit — the module migrates once "
 "P(F_[n] = fail | e) exceeds the ratio C_[mig] / C_[fail] — so the effective threshold is "
 "derived per module from the relative cost of failing and moving. This directly answers the "
 "threshold-selection difficulty acknowledged by {rajab2021|t} and {gupta2025|t}, because there "
 "is no longer a single global threshold to select."),

("h2", "D.", "Novelty 3: Criticality-Aware Selective Replication"),

("p",
 "Prior schemes provision a backup for every protected task {rajab2021;naim2016} or pre-select "
 "a backup node for every placed microservice {gupta2025}. The standby capacity so reserved is "
 "consumed whether or not a failure occurs, which {gupta2025|t} identify as a standing cost of "
 "their design. The proposed framework instead treats redundancy as a resource to be allocated "
 "where it changes the outcome. Each module is assigned a criticality weight reflecting the "
 "consequence of its loss, and the controller computes a risk score"),

("eq", "R(m)  =  P(F_[n] = fail | e) · κ(m)", "(4)"),
("where", [
    ("R(m)",
     "the risk score of module m: the expected consequence of its loss over the coming horizon, "
     "and the quantity on which its redundancy tier is decided."),
    ("P(F_[n] = fail | e)",
     "the posterior failure probability from (1) for the node n that currently hosts m."),
    ("κ(m)",
     "the criticality weight of m, on a scale of 0 to 1, reflecting the consequence of losing "
     "it — near 1 for a cardiac arrhythmia classifier, near 0 for a best-effort module. It is "
     "the same criticality that enters C_[fail](m) in (3)."),
]),

("p",
 "Modules are then assigned to one of three redundancy tiers by comparing R(m) against two "
 "boundaries τ_[high] and τ_[mid]. High-risk, high-criticality modules, for which R(m) ≥ "
 "τ_[high] — a cardiac arrhythmia classifier on a node the network judges fragile — receive a "
 "warm standby on a pre-selected low-risk node, with state synchronised so that failover is "
 "immediate. Modules of intermediate risk, τ_[mid] ≤ R(m) < τ_[high], receive periodic state "
 "checkpointing without a dedicated standby, so that recovery requires re-instantiation but not "
 "re-execution. Low-risk, best-effort modules, for which R(m) < τ_[mid], receive no redundancy "
 "and are simply restarted if they fail. Because tier assignment is recomputed as posteriors "
 "change, standby capacity follows risk dynamically rather than being pinned at placement time, "
 "and the aggregate redundancy overhead is bounded by a configurable capacity budget B that the "
 "controller respects when admitting modules to the highest tier: where the budget is "
 "exhausted, the modules of greatest R(m) hold the warm standbys."),

("h2", "E.", "End-to-End Workflow"),
("p",
 "The framework operates as a continuous control loop over discrete decision epochs, summarised "
 "in Fig. 3:"),

("num", "Telemetry agents on every fog node report the observable indicators to the controller."),
("num",
 "The pre-processing stage aligns, cleans and discretises the measurements into the ordinal "
 "states of the evidence variables, flagging any channel that failed to report."),
("num",
 "The BBN inference engine computes the posterior failure probability for every node by (1), "
 "marginalising over unobserved evidence, and records the dominant causal explanation."),
("num",
 "For newly arriving modules, the risk-aware placement policy ranks candidate nodes by the "
 "multi-objective utility (2) and places each module on the highest-scoring feasible node."),
("num",
 "For running modules, the expected-loss rule (3) is evaluated; modules for which the expected "
 "failure loss exceeds the migration cost are migrated pre-emptively to the best alternative "
 "node."),
("num",
 "The replication manager recomputes the risk scores (4), assigns modules to redundancy tiers "
 "and provisions or releases standby capacity subject to the redundancy budget."),
("num",
 "Actuated decisions are applied to the fog layer; modules that no fog node can safely host are "
 "offloaded to the cloud."),
("num",
 "Observed outcomes are compared against predictions, the conditional probability tables are "
 "updated by Bayesian parameter learning, and the revised tables are synchronised with the "
 "cloud repository before the next epoch begins."),

("fig", "figures/paper_fig3_flow.png", 4.80, 4.83),
("cap", "Fig. 3.",
 "End-to-end control loop of the proposed framework over one decision epoch. Steps 1-3 perform "
 "inference, steps 4-6 convert the posterior into placement, migration and replication "
 "decisions, step 7 actuates those decisions on the fog layer and offloads whatever the fog "
 "cannot host, and step 8 closes the loop by updating the conditional probability tables from "
 "observed outcomes and synchronising them with the cloud repository, so that the model becomes "
 "progressively better calibrated to the deployment it governs."),
]

BODY += [

# ======================= IV. EXPERIMENTAL SETUP =======================
# Table III (datasets) and the whole of the former IV.C "Device Instantiation and
# Parameterisation" with Table IV were removed on the guide's instruction; the one
# on-topic paragraph of IV.C -- the mapping from evidence variables to simulator
# quantities -- was condensed and moved into III.B above.
("h1", "IV.", "Experimental Setup and Dataset Description"),
("h2", "A.", "Simulation Environment"),

("p",
 "The framework is being implemented as an extension to iFogSim2 {mahmud2022}, the widely used "
 "discrete-event simulator for edge and fog resource management, which builds on CloudSim and "
 "on the original iFogSim toolkit {gupta2017}. iFogSim2 is selected because it already provides "
 "the three capabilities the evaluation requires: mobility support driven by real and synthetic "
 "location traces, microservice orchestration with horizontal scaling and service discovery, "
 "and dynamic clustering of fog nodes. The proposed controller is realised as a new placement "
 "logic registered with the microservices controller, so that it can be compared against the "
 "simulator's existing placement policies under identical topologies and workloads. Because "
 "iFogSim2 does not model node failure natively, a fault-injection harness is being added that "
 "fails nodes according to configurable stochastic processes conditioned on their simulated "
 "load, energy and link state, thereby generating both the failure events to be predicted and "
 "the ground truth against which predictions are scored."),

("h2", "B.", "Dataset Description"),

("p",
 "Three categories of data drive the evaluation. The physical topology is taken from the "
 "Melbourne central business district edge-resource dataset distributed with iFogSim2, which "
 "enumerates 131 edge and fog resources with their geographic coordinates and parent-child "
 "relationships across a four-level hierarchy of cloud data centre, proxy server, gateway and "
 "end user. User mobility is supplied both by a real trace of 87 recorded positions and by five "
 "synthetic traces of 100 positions each, generated by the simulator's random mobility "
 "generator; mobility matters here because handover is one of the evidence variables in the "
 "network and a driver of load redistribution. Application workloads comprise three models: a "
 "cardiovascular health-monitoring application structured as a client module feeding two "
 "downstream analysis microservices, an intelligent surveillance application in which a motion "
 "detector, object detector and object tracker drive a pan-tilt-zoom actuator, and a "
 "virtual-reality gaming application in which electroencephalogram data is classified and fed "
 "to a global coordinator. A configuration file accompanying the microservice application "
 "supplies ten distinct application profiles specifying per-microservice processing "
 "requirements, inter-module network loads, per-hop latencies, throughput and budget "
 "constraints, and these parameters define the operating envelope from which the evidence "
 "variables are derived."),

("p",
 "It should be stated plainly that no public failure-trace dataset exists for a fog topology of "
 "this kind. Failure events are therefore synthesised by the injection harness described above, "
 "and the conditional probability tables are seeded from priors reported in the reliability "
 "literature {fenton2007;weber2012;pang2025} before being adapted online. Validating the "
 "framework against a real fog failure trace, should one become available, is identified as "
 "future work."),

("h2", "C.", "Evaluation Metrics and Baselines"),

("p",
 "The framework will be assessed along two axes. Prediction quality will be measured by "
 "precision, recall and F1 score of failure prediction within the look-ahead horizon, by the "
 "calibration of the posterior against observed failure frequency, and by lead time — the "
 "interval between a prediction crossing the decision rule and the actual failure — since a "
 "prediction that arrives too late to act upon has no operational value. System quality will be "
 "measured by mean and tail end-to-end application loop delay, deadline-violation ratio for "
 "critical loops, total energy consumption across cloud, fog and end-device tiers, network "
 "usage, execution cost, service downtime per injected failure, and redundancy overhead "
 "expressed as reserved standby capacity. Robustness will be probed by withholding telemetry "
 "channels at inference time to quantify the degradation of prediction quality under partial "
 "evidence."),

("p",
 "Four baselines will be compared under identical conditions: cloud-only placement; the default "
 "edgeward placement of iFogSim2 with no fault tolerance; reactive checkpointing with "
 "primary-backup replication in the manner of {rajab2021|t}; and threshold-based proactive "
 "migration in the manner of FTAPA {gupta2025}, which is the closest competitor and the most "
 "informative comparison, since it isolates the contribution of probabilistic inference from "
 "the contribution of acting early."),

# ======================= V. CONCLUSION =======================
("h1", "V.", "Conclusion and Future Work"),

("p",
 "This paper has argued that fault tolerance in fog computing remains predominantly reactive, "
 "and that the proactive mechanisms so far reported reduce an inherently uncertain judgement to "
 "a fixed threshold on a single metric. A survey of five representative fog fault-tolerance "
 "systems found that none quantifies the probability of imminent node failure, none reasons "
 "under partial telemetry, and none modulates redundancy by predicted risk. Drawing on an "
 "established body of work applying Bayesian Belief Networks to reliability and dependability, "
 "the paper has proposed a framework in which a causal BBN infers the posterior probability of "
 "fog node failure from noisy and possibly incomplete evidence, a risk-aware utility and an "
 "expected-loss decision rule convert that posterior into placement and pre-emptive migration "
 "decisions, and a criticality-aware scheme allocates standby capacity in proportion to risk "
 "and consequence. The architecture, its block diagram and its workflow have been presented, "
 "together with the simulation environment and the datasets on which it will be evaluated."),

("p",
 "Work in Phase II will proceed in four stages: construction and expert validation of the BBN "
 "structure and its initial conditional probability tables; implementation of the inference "
 "engine and the fault-injection harness within iFogSim2; implementation of the risk-aware "
 "placement, expected-loss migration and selective replication policies as an iFogSim2 "
 "placement logic; and comparative evaluation against the four baselines across the metrics "
 "enumerated in Section IV. Longer-term extensions include a dynamic Bayesian network "
 "formulation to model the temporal evolution of node degradation explicitly, and distributed "
 "inference across clustered fog nodes to remove the controller as a single point of failure."),
]
