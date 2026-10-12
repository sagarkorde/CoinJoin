"""
Content for the three per-reviewer response letters (manuscript jcp-4517330).

Separated from the builder so that the wording can be edited without touching
the Word mechanics. Every figure quoted here is taken from the combined
response letter, which was itself written from the stored experiment outputs
in research/results/.

Reviewer 1's General Evaluation entries are the ticks recorded on the SuSy
reply page (MDPI _ Reply review report_Reviewer 1.pdf). Reviewers 2 and 3
returned no such page, so their entries are inferred from the review report
forms: MDPI renders a comment box under a question only where the reviewer
wrote something, and in Reviewer 1's verified case every commented question
was answered No and every uncommented one Yes. Flagged in VERIFY_BEFORE_SUBMIT.
"""
from __future__ import annotations

TITLE = ("MixTrace: Auditing Graph-Based Bitcoin Forensics Through "
         "Leakage-Free Evaluation and External CoinJoin Validation")
REPO = "https://github.com/sagarkorde/CoinJoin"
ZENODO = "https://doi.org/10.5281/zenodo.23275812"

# The question set is the one the JCP review report form actually puts to
# reviewers, not the generic set in MDPI's example file. The two differ, and
# only the former can be matched to the ticks the reviewers recorded.
QUESTIONS = [
    "Does the title describe the article's topic with sufficient precision?",
    "Does the introduction provide a comprehensive yet concise overview about "
    "the state of knowledge in the area of research?",
    "Is the research design appropriate and are the methods adequately "
    "described?",
    "Are the results presented clearly and in sufficient detail, are the "
    "conclusions supported by the results and are they put into context "
    "within the existing literature?",
    "Are all of the cited references relevant to the research?",
    "Does this article provide a relevant contribution to the scientific "
    "discussion of this topic?",
    "Are all figures and tables clear and well-presented?",
]

# --------------------------------------------------------------------------
# Reviewer 1
# --------------------------------------------------------------------------

R1_SUMMARY = [
    "Thank you very much for taking the time to review this manuscript. The "
    "report is the reason this revision has the shape it does, and we would "
    "rather describe that plainly than present it as a polishing pass.",

    "Three of the comments concerned leakage and circularity. Two were "
    "correct, and the audit they prompted found the problem to be worse than "
    "the report described; the results they affected have been withdrawn "
    "rather than defended. One, the Taproot limitation, rested on a premise "
    "we have had to contradict, and the measurement that obliges us to do so "
    "is given in Response 5. It is the only point in this letter on which we "
    "do not agree with the reviewer.",

    "Rather than editing the manuscript around the comments, the study was "
    "re-executed from the raw data. Eighteen experiments were run. Every "
    "number in the revised manuscript is produced by a script in "
    "research/experiments/, written to a file in research/results/, and tied "
    "to a dated commit recorded in research/EXPERIMENT_LOG.md at " + REPO +
    ". The verification data behind the new external validation, together "
    "with the raw block-explorer responses it was built from, are deposited "
    "at " + ZENODO + ".",

    "Two conclusions from round one do not survive. The two engineered graph "
    "features are withdrawn as a label leak. The claim that self-supervised "
    "pre-training is the primary performance driver is also withdrawn, "
    "because across twenty seeds a plain GraphSAGE baseline exceeds the "
    "fused model and every graph model trails a Random Forest on the same "
    "features. Both withdrawals are set out in Section 5 of this letter.",

    "All revisions are marked in Manuscript_Revised_Highlighted.docx; "
    "Manuscript_Revised_Clean.docx is the file for publication.",
]

R1_GENERAL = [
    ("No",
     "Addressed in Response 7. The title is now “" + TITLE + "”. It "
     "names the two tasks separately and no longer implies that the graph "
     "attention model detects CoinJoin transactions. The abstract and the "
     "contribution statement were rewritten to match."),
    ("Yes",
     "Thank you. No change was requested on this point. The introduction was "
     "edited only where the revision required it: the contribution list now "
     "describes three independently evaluated modules rather than an "
     "integrated pipeline, and the Taproot framing follows the corrected "
     "measurement reported in Response 5."),
    ("No",
     "Addressed in Responses 1, 2, 3, 4, 8 and 10. The two leaking features "
     "are withdrawn; the transductive and inductive pre-training regimes are "
     "trained and reported side by side; the screening label is stated as "
     "the exact rule it is; the cluster-to-class mapping is fitted on "
     "training data and frozen; threshold selection is enforced in a single "
     "code path; and the architecture is renamed to the fusion it actually "
     "is."),
    ("No",
     "Addressed in Responses 5, 6, 11 and 12. The Taproot claims now rest on "
     "direct measurement rather than on a proxy, the integrated-pipeline "
     "framing is removed, the statistical basis is twenty seeds plus a "
     "test-set bootstrap, and the screening result is described throughout "
     "as a conservative operating point. Two further conclusions are "
     "withdrawn outright, as set out in Section 5 below."),
    ("No",
     "The reference cited for DBSCAN was indeed mismatched: it was Saruladha "
     "and Likhitha on the Modified ROCK algorithm for categorical data, "
     "which is a different method. It is corrected to Ester, Kriegel, Sander "
     "and Xu (1996), the original DBSCAN paper. Every remaining citation was "
     "then checked against the claim it is used to support, and the "
     "reference list was updated where the revision changed the claim."),
    ("Yes",
     "Thank you. The contribution is now stated more narrowly than in round "
     "one, and the revision adds a validation that did not exist before: "
     "4,491 transactions in the corpus are confirmed as CoinJoin outputs "
     "against published protocol rules using blockchain data, independently "
     "of the heuristic under evaluation."),
    ("No",
     "Addressed in Response 9. Every figure drawn from the superseded "
     "inference-only ablation has been regenerated under the final protocol, "
     "and the single-seed figures are either replaced by seed-aggregated "
     "versions with variance bands or labelled explicitly as single-run "
     "diagnostics."),
]

R1_POINTS = [
    (
        "Potential label and temporal leakage in the engineered Elliptic "
        "features. The Illicit Neighbor Ratio explicitly depends on "
        "neighbouring labels, while Shortest Path to Illicit is computed from "
        "confirmed illicit seed nodes and reaches a very high correlation "
        "with the target (r = −0.9182). The authors should demonstrate "
        "unambiguously that validation and test labels are never used in "
        "constructing either feature. In addition, if shortest-path "
        "computation traverses edges or nodes from future time steps, the "
        "resulting feature may introduce look-ahead information even when "
        "test labels themselves are withheld. A leakage-free ablation should "
        "therefore be reported using: (i) the original 165 Elliptic features, "
        "(ii) + Illicit Neighbor Ratio, (iii) + Shortest Path to Illicit, and "
        "(iv) + both engineered features.",
        [
            "Thank you for pointing this out. We agree with this comment, and "
            "we cannot give the demonstration the reviewer asks for, because "
            "the labels do enter the features. Both are withdrawn.",

            "What the audit found. shortest_path_to_illicit is a "
            "breadth-first distance computed from every confirmed illicit "
            "node, and each illicit node is its own source, so the feature is "
            "exactly zero for all 4,545 illicit nodes in the labelled set and "
            "non-zero for all 42,019 licit ones. The decision rule "
            "“shortest path equals zero implies illicit” therefore "
            "has precision 1.000, and it applies on the test split. Illicit "
            "Neighbour Ratio reads neighbouring labels directly, as the "
            "reviewer says. Removing the self-seeding does not repair either "
            "feature, for the reason given below.",

            "Before auditing we reproduced the round-one computation exactly, "
            "recovering the reported correlation of r = −0.9182 bit for "
            "bit, which establishes that the audit targets the same code path "
            "rather than a reconstruction of it.",

            "The four-way ablation is now Table 1 of the revised manuscript. "
            "Test split, mean over five seeds, threshold selected on "
            "validation and frozen before scoring:",

            "(i) 165 distributed features: Random Forest F1 0.6212, PR-AUC "
            "0.5431; Hist. Gradient Boosting F1 0.6126, PR-AUC 0.5561. "
            "(ii) plus Illicit Neighbour Ratio: 0.6333 and 0.6158; 0.6811 and "
            "0.6794. (iii) plus Shortest Path to Illicit: 0.9990 and 1.0000; "
            "1.0000 and 1.0000. (iv) plus both: 0.9953 and 1.0000; 1.0000 and "
            "1.0000.",

            "A perfect score under two unrelated inductive biases is the "
            "signature of target leakage, not of a strong feature.",

            "On the look-ahead question. The concern cannot arise in the form "
            "described, because the Elliptic edge set contains no "
            "cross-time-step edges at all: 0 of 234,355. But that same fact "
            "removes any possibility of repair. The graph is a disjoint union "
            "of forty-nine per-step snapshots, so no path connects a "
            "training-period node to a test-period node, and under a strict "
            "protocol both features become constant on validation and test. "
            "No leakage-free variant of either feature can exist on this "
            "graph. The argument is set out in Section 3.2.2 under “No "
            "leakage-free variant can exist”.",

            "Both features are withdrawn from the manuscript, from the "
            "abstract and from the contribution statement. Label-free "
            "topological replacements were developed and tested, and they did "
            "not recover the lost performance; that negative result is "
            "reported rather than omitted. Experiments E01 and E02.",
        ],
    ),
    (
        "The transductive nature of DGI pre-training should be explicitly "
        "acknowledged. DGI is pre-trained over the full graph, including all "
        "203,769 nodes, whereas the paper describes the chronological split "
        "as preventing temporal leakage. Since future validation/test nodes "
        "contribute unlabeled features and topology to representation "
        "learning, the current experiment is transductive rather than "
        "strictly prospective or inductive. The authors should make this "
        "distinction explicit and, ideally, provide an additional experiment "
        "in which DGI is trained only on the training-period graph.",
        [
            "Agree. The distinction is now explicit in the manuscript, and "
            "the additional experiment the reviewer suggests has been run, so "
            "the question is settled empirically rather than only "
            "acknowledged.",

            "Both regimes are trained and reported side by side across twenty "
            "seeds. In the transductive regime the encoder sees the whole "
            "graph, as in round one. In the inductive regime the encoder is "
            "fitted on the training-period subgraph alone, and embeddings for "
            "later nodes are produced by a forward pass of the frozen "
            "encoder, so no validation or test node contributes to "
            "representation learning at any point.",

            "The two are indistinguishable. Inductive fusion reaches a mean "
            "illicit-class F1 of 0.5197 and transductive fusion 0.5186, a "
            "difference of 0.0011, with a paired t-test over twenty seeds "
            "giving t = 0.24 and p = 0.815. Restricting pre-training to the "
            "training period costs essentially nothing, so the benefit "
            "reported in round one is not an artefact of letting the encoder "
            "observe future nodes.",

            "The manuscript nevertheless keeps the distinction explicit, "
            "because the two settings answer different questions, and the "
            "inductive regime is now the reference arm against which the "
            "other systems are compared. Experiment E05; results in Figure 4 "
            "with the accompanying model comparison table, and in Table 15.",
        ],
    ),
    (
        "The status of the CoinJoin labels requires clarification. The "
        "author-curated dataset uses an is_coinjoin_like label derived from "
        "structural heuristics rather than independently confirmed CoinJoin "
        "ground truth. The manuscript should specify the exact rules used to "
        "generate these labels and clarify whether any of the same variables "
        "subsequently used by DBSCAN were also used to construct the target. "
        "Otherwise, the experiment risks circular evaluation. Independent "
        "validation against externally confirmed transactions from known "
        "mixing services would substantially strengthen this part of the "
        "study.",
        [
            "Thank you. We agree, and the revision goes beyond clarification: "
            "the rule is now stated exactly, the circularity is quantified "
            "and shown to be irreducible by feature selection, and the "
            "independent validation the reviewer suggests has been carried "
            "out. This is the largest addition to the revision, and it exists "
            "because of this comment.",

            "The exact rule. The round-one manuscript described "
            "is_coinjoin_like as a heuristic without stating it. We recovered "
            "it: a depth-three decision tree reproduces the field with "
            "accuracy 1.00000000, and the identity is_coinjoin_like == "
            "(input_count >= 4) AND (output_count >= 4) holds for every one "
            "of the 5,884,387 transactions in the corpus. The rule is now "
            "given in the manuscript under “Provenance of the Screening "
            "Label” and the term “ground truth” is removed "
            "throughout. Experiment E06.",

            "The circularity is confirmed, and it cannot be removed by "
            "dropping columns. Both counts were among the features the "
            "round-one detector clustered on, so the reviewer's concern holds "
            "as stated. We then tested whether removing those columns repairs "
            "the evaluation, and it does not: the counts are recoverable by "
            "arithmetic from the remaining columns along at least two routes. "
            "Transaction size is an affine function of the counts, and "
            "regressing the counts on the size family recovers the Bitcoin "
            "serialisation constants directly (vsize approximately 20.5 + "
            "90.75 n_in + 35.15 n_out, R-squared 0.817; weight is exactly "
            "four times vsize). More decisively, avg_input_value is "
            "total_input_value divided by input_count, so the quotient "
            "recovers input_count exactly for 99.96 percent of rows and "
            "output_count for 99.98 percent across 500,000 sampled "
            "transactions, and the "
            "rule reconstructed from those quotients agrees with the "
            "distributed label for 100.0000 percent of sampled transactions. "
            "The manuscript therefore reports a gradient across progressively "
            "stricter feature regimes, in Table 2, rather than claiming that "
            "any single regime is clean.",

            "Independent validation against known mixing services. Detection "
            "rules published with Dumplings, the reference tool used in the "
            "measurement literature to extract Wasabi, Whirlpool and "
            "JoinMarket transactions from mainnet, were applied to per-output "
            "values retrieved from public block explorers. The corpus records "
            "aggregate and mean output values but not individual amounts, and "
            "every rule turns on individual amounts, so nothing can be "
            "confirmed from the corpus alone. Candidates were selected by the "
            "necessary conditions the columns can express, and each was then "
            "tested against the full protocol rule using blockchain data. All "
            "6,209 candidates were retrieved, with no failures. Table 10 "
            "reports the outcome: Whirlpool-shaped, 4,137 verified and 3,839 "
            "confirmed, 92.8 percent; WabiSabi-shaped, 1,522 verified and 652 "
            "confirmed, 42.8 percent; a random sample of 400 labelled "
            "positives, none confirmed; a random sample of 150 labelled "
            "negatives, none confirmed.",

            "The 3,839 confirmed Whirlpool rounds distribute across the four "
            "denomination pools as 1,753, 1,103, 780 and 203, matching "
            "reported usage in which the smallest pool is busiest. That is a "
            "consistency check the procedure was not designed to pass.",

            "What this says about the label. Stratified over all 110,352 "
            "labelled transactions, the screening label attains a precision "
            "of 4.07 percent, with a 95 percent upper bound of 4.97 percent, "
            "at near-total recall (Table 11). It admits roughly twenty-four "
            "transactions for every CoinJoin it contains, and the manuscript "
            "now says so in those terms. Experiments E14 and E14b; the raw "
            "block-explorer responses are in the Zenodo deposit.",
        ],
    ),
    (
        "Possible contamination in DBSCAN cluster-to-class assignment. The "
        "manuscript states that “cluster CoinJoin labels are assigned by "
        "majority vote over the ground truth labels of the cluster "
        "members.” This is potentially problematic if labels from the "
        "same transactions later used for evaluation contribute to the "
        "cluster-class mapping. The authors should clearly separate training "
        "and testing: clusters or cluster-to-class mappings should be derived "
        "from training data only, frozen, and then evaluated on an untouched "
        "test partition.",
        [
            "Agree, and corrected exactly as proposed. Clusters are now "
            "fitted on training-period transactions only, the cluster-to-"
            "class mapping is frozen, and test transactions are assigned by "
            "nearest fitted centroid without contributing to the mapping.",

            "The audit also found a problem the comment did not reach: round "
            "one had no temporal separation on this corpus at all. A "
            "chronological split has been introduced, with training to "
            "2024-06, validation to 2024-12 and test on 2025. The majority "
            "vote was additionally replaced by a positive-rate threshold "
            "selected on the validation split, because majority vote is "
            "ill-defined for the small clusters this corpus produces.",

            "Results under the corrected protocol are in Table 9, reported "
            "alongside a supervised reference and for both the full and the "
            "restricted feature regimes, so that the effect of the correction "
            "and the effect of the feature regime can be read separately. "
            "Experiment E08.",
        ],
    ),
    (
        "Taproot-related claims should be moderated. The manuscript "
        "explicitly states that the author-curated corpus contains no "
        "confirmed P2TR transactions and that multi-input P2WPKH "
        "transactions are used as proxies. Consequently, terminology "
        "suggesting empirical “Taproot-aware” performance should be "
        "avoided. The present evidence supports a script-agnostic structural "
        "approach under pre-Taproot proxy conditions, not validated "
        "performance on live Taproot CoinJoin transactions.",
        [
            "Thank you for raising this. We have to answer it differently "
            "from the others, because the premise does not hold, and we ask "
            "the reviewer to re-examine the point against the measurement "
            "below.",

            "The statement that the corpus contains no confirmed P2TR "
            "transactions came from six boolean script-type columns. Those "
            "six columns are identically False for every one of the 5,884,387 "
            "rows. They do not report the absence of Taproot; they report "
            "nothing at all about any script type, and the round-one "
            "manuscript read a broken column as a finding. The error was "
            "ours, not the reviewer's.",

            "Parsing the node-reported script descriptors instead gives the "
            "composition in Table 3. P2TR accounts for 24.10 percent of "
            "outputs. 2,494,168 transactions, or 42.39 percent of the corpus, "
            "involve Taproot, and 71.94 percent of CoinJoin-like candidates "
            "do. The corpus runs from 2022-07-13 to 2025-07-01 and is "
            "post-Taproot throughout.",

            "The proxy analysis has therefore been replaced by direct "
            "measurement on real P2TR data, and the Taproot statements in the "
            "manuscript are stronger than in round one rather than moderated. "
            "We would not make that move without the measurement: the parsing "
            "script and its output are in the repository, and the correction "
            "is reported in the manuscript under “Correction to the "
            "reported script coverage”. Experiment E07.",

            "The address-count columns were later found to be broken in the "
            "same way. The pattern, a derived column whose name describes a "
            "quantity it does not hold, is now discussed in its own right "
            "under Threats to Validity, because it is not detectable by "
            "inspecting summary statistics and it affected two independent "
            "claims in round one.",
        ],
    ),
    (
        "The three components are better characterised as a multi-module "
        "framework than an integrated pipeline. Figure 1 shows structural "
        "clustering, CST linking, and illicit classification as parallel "
        "modules that are evaluated independently. They operate on different "
        "tasks and partly different datasets, and there is no demonstrated "
        "end-to-end information flow from one module to the next. The "
        "contribution statement and conclusion should therefore avoid "
        "implying that the three components form a jointly optimised "
        "end-to-end pipeline.",
        [
            "Agree, without reservation. The contribution statement, the "
            "abstract, Figure 1 and the conclusion are rewritten to describe "
            "three independently evaluated modules. No claim of joint "
            "optimisation or of end-to-end information flow between the "
            "modules remains anywhere in the manuscript.",

            "The point is also stated against the study rather than only "
            "removed from it: “The modules are evaluated separately” "
            "is now one of the entries in the Limitations section, with the "
            "reason, which is that no single dataset carries all three label "
            "types.",
        ],
    ),
    (
        "The title should distinguish CoinJoin analysis from general "
        "illicit-transaction detection. The GAT component predicts illicit "
        "versus licit transactions on Elliptic, whereas the structural "
        "clustering component identifies CoinJoin-like transactions. The "
        "manuscript does not establish that the illicit transactions "
        "classified in Elliptic are specifically CoinJoin transactions. A "
        "more precise title could therefore distinguish “CoinJoin "
        "analysis” from “illicit Bitcoin transaction detection.”",
        [
            "Agree. The title is now “" + TITLE + "”.",

            "It names the two tasks separately, keeps the system name for "
            "continuity with the submitted record, and no longer implies that "
            "the graph attention model detects CoinJoin transactions. The "
            "abstract was rewritten to the same effect: illicit-transaction "
            "detection on the Elliptic Data Set and CoinJoin screening on the "
            "author-curated corpus are presented as separate evaluations, on "
            "separate datasets, with separate labels and separate "
            "conclusions.",
        ],
    ),
    (
        "Threshold-selection protocols are inconsistent across the "
        "manuscript. Algorithm 1 defines a validation-calibrated threshold "
        "and reports θ* = 0.9397, whereas the five-seed results in Table "
        "1 are later described as using a fixed 0.5 threshold. These should "
        "be harmonised. A standard protocol would be to select the threshold "
        "on validation data independently for each training run and then "
        "freeze it before test evaluation.",
        [
            "Agree, and the protocol the reviewer describes is now the one "
            "the study uses. It is enforced in code rather than stated in "
            "prose: the threshold maximises F1 on the validation split, is "
            "selected independently for each training run, and is frozen "
            "before the test split is scored.",

            "research/src/metrics.py is the only sanctioned path to a "
            "threshold in the codebase, and it also reports threshold-free "
            "metrics, PR-AUC and ROC-AUC, alongside every thresholded one, so "
            "the inconsistency the reviewer found cannot recur silently.",

            "The value 0.9397 does not appear in the revised manuscript. "
            "Selected thresholds are per-run, and are reported where they "
            "carry meaning, for example 0.8292 for the primary seed of the "
            "fused model, which is the operating point at which Table 6 "
            "profiles the four outcome groups.",
        ],
    ),
    (
        "The GAT-without-DGI ablation should be made fully consistent. The "
        "manuscript alternately describes the ablation as a separate "
        "128-dimensional GAT architecture and as a model in which the DGI "
        "skip representation is replaced by zeros from the beginning of "
        "training. Furthermore, several figures still correspond to an "
        "earlier inference-only ablation that the manuscript itself "
        "acknowledges as superseded. All headline figures should be "
        "regenerated using the final, genuinely DGI-free model reported in "
        "Table 1.",
        [
            "Agree. The no-DGI arm is now trained from scratch with the fused "
            "slot zeroed from the first epoch, not ablated at inference, and "
            "every figure has been regenerated from the final protocol. No "
            "figure from the superseded run survives in the manuscript or in "
            "the supplementary material.",

            "Measured against that corrected ablation over twenty seeds, the "
            "fused model exceeds it by 0.049 F1 in the transductive regime "
            "and by 0.063 in the inductive regime, with paired t-tests of "
            "3.26 (p = 0.0041) and 4.35 (p = 0.00034). The variance claim "
            "also survives the correction: the cross-seed standard deviation "
            "falls from 0.0594 to 0.0303, approximately a halving. Both are "
            "reported in Table 15.",

            "Separately, and in line with Reviewer 3, the figures that "
            "previously showed a single representative seed are now either "
            "seed-aggregated with variance bands or labelled explicitly as "
            "single-run diagnostics, so that no figure in the manuscript can "
            "be read as the headline statistic.",
        ],
    ),
    (
        "The terminology “DGI pre-training followed by GAT "
        "fine-tuning” should be reconsidered. The architecture appears "
        "to use a frozen DGI/GCN embedding concatenated with a separately "
        "trained GAT representation. The GAT itself is not initialised from "
        "DGI weights and the DGI encoder is not subsequently fine-tuned "
        "through the GAT objective. This is more accurately described as "
        "fusion of a frozen self-supervised DGI representation with a "
        "supervised GAT representation.",
        [
            "Agree, and the reviewer's reading of the code is correct. The "
            "DGI encoder is frozen, the GAT is trained separately, and the "
            "two representations are concatenated. No DGI weight is updated "
            "by the supervised objective.",

            "The manuscript now describes this as fusion of a frozen "
            "self-supervised representation with a supervised one, and the "
            "phrase “pre-training followed by fine-tuning” is "
            "removed from the title of the architecture section, the method "
            "description, the abstract and the conclusion.",

            "The implementation class has been renamed FrozenDGIGATFusion so "
            "that the code and the manuscript cannot drift apart on this "
            "point again.",
        ],
    ),
    (
        "Statistical significance should be interpreted more cautiously. Five "
        "random seeds quantify stochastic training variability on one fixed "
        "dataset partition; they do not constitute five independent samples "
        "of generalisation performance. The paired t-tests are therefore "
        "informative about seed-level consistency but should not be "
        "interpreted as comprehensive statistical evidence of "
        "population-level superiority. A bootstrap analysis of test-set "
        "metric differences or a temporally blocked bootstrap would provide a "
        "stronger uncertainty assessment.",
        [
            "Agree. Seeds are increased from five to twenty, and the "
            "test-set bootstrap the reviewer suggests has been added: paired "
            "bootstrap resampling of the labelled test nodes, with both "
            "systems scored on the same resampled nodes, reported as a "
            "difference with a percentile confidence interval. Table 16 gives "
            "the result over 5,000 resamples of the 8,841 labelled test "
            "nodes.",

            "Seed variance and test-set variance are now reported as the "
            "distinct quantities they are, each with an explicit scope, and "
            "the paired t-tests are described only as evidence of seed-level "
            "consistency, in the terms the reviewer sets out.",

            "The two analyses disagree in one place, and the manuscript says "
            "so rather than reporting the more convenient one. The bootstrap "
            "places the fused model above GraphSAGE by 0.0430 with an "
            "interval excluding zero; the seed-level test places GraphSAGE "
            "above the fused model by 0.0270. The subsection “The Two "
            "Analyses Disagree, and That Is Informative” sets out why "
            "both can be correct and what each is entitled to claim.",
        ],
    ),
    (
        "The DBSCAN result represents a substantial operating-point trade-off "
        "rather than an unqualified improvement. Structural DBSCAN reduces "
        "FPR from 3.72% to 0.75%, but simultaneously produces an FNR of 70% "
        "and reduces F1 from 0.870 to 0.441. The manuscript appropriately "
        "discusses the forensic motivation for high precision, but this "
        "result should consistently be described as a conservative "
        "operating-point trade-off rather than overall superior detection "
        "performance. A precision–recall or FPR–FNR curve across "
        "parameter settings would be useful.",
        [
            "Agree. The result is described as a conservative operating point "
            "rather than as superior detection, consistently: in the "
            "abstract, in the results and in the discussion.",

            "The curves the reviewer asks for are now reported across the "
            "full range of clustering radii and decision thresholds, in "
            "Figure 9. The false-negative rate stays above 0.3 throughout the "
            "swept range, so the trade-off is a property of the method and "
            "not of the particular setting chosen in round one.",

            "The manuscript also now states what screening output can and "
            "cannot support in forensic practice, under “What screening "
            "output can and cannot support”, since a label whose "
            "precision against confirmed CoinJoin transactions is 4.07 "
            "percent is an indication for further work and not an "
            "attribution.",
        ],
    ),
]

R1_ENGLISH = (
    "The review report records “The English could be improved to more "
    "clearly express the research”, and raises no separate point on the "
    "language.",
    [
        "Thank you. A full editorial pass has been made over the whole "
        "manuscript. The informal expressions identified by the other reports "
        "(“sort of”, “kind of”, “so yeah”) are "
        "removed, and the corrupted sentence in the round-one Section 5.1 is "
        "corrected.",

        "The larger part of the problem was not style. Several passages "
        "claimed more than the evidence supported, and the clarity improves "
        "chiefly because those claims are now stated at the strength the "
        "results allow. The revised manuscript is approximately 20,300 words "
        "against approximately 12,400 in round one, and the added material "
        "was written to the same standard and read back for concision.",
    ],
)

R1_EXTRA = [
    "Where to find things. The clean file is Manuscript_Revised_Clean.docx "
    "and the marked copy is Manuscript_Revised_Highlighted.docx, in which 267 "
    "of 309 substantive paragraphs are highlighted. The proportion is high "
    "because the study was re-executed rather than edited; the unmarked text "
    "is largely the related-work and ethics material, which the revision did "
    "not need to change.",

    "Reproducibility. Every experiment has a script in research/experiments/, "
    "an output file in research/results/, and a dated commit recorded in "
    "research/EXPERIMENT_LOG.md at " + REPO + ". The verification data, the "
    "raw block-explorer responses behind it, the collection manifest and the "
    "checksum list are deposited at " + ZENODO + ", and the Data Availability "
    "Statement points to both.",

    "Two conclusions from round one are withdrawn. The engineered graph "
    "features are withdrawn as a label leak, as set out in Response 1. The "
    "claim that self-supervised pre-training is the primary performance "
    "driver is also withdrawn: across twenty seeds a plain GraphSAGE baseline "
    "reaches F1 0.5467 against 0.5197 for the fused model, using 29,570 "
    "parameters instead of 131,588, one ninth of the training time and one "
    "sixth of the peak memory, and every graph model trails a Random Forest "
    "on the same features at 0.6212. The narrower claim the evidence does "
    "support, that the frozen representation improves the graph attention arm "
    "specifically in both mean and variance, is what the manuscript now "
    "states.",

    "A finding that came out of the review rather than being asked for. "
    "Broken out by time step, as Reviewer 2 requested, the classifier "
    "achieves a mean illicit-class F1 of 0.8582 across steps 35 to 42 and "
    "0.0283 across steps 43 to 49, with recall falling from 0.789 to 0.000 at "
    "step 43, the dark-market shutdown. The round-one test window lies almost "
    "entirely inside the collapsed regime. This also explains the appeal of "
    "the leaked feature: after the shock, nothing learned from the training "
    "period still works, so a feature that re-encodes the label is the only "
    "thing holding the metric up.",

    "Authorship. A change of authorship request accompanies this revision, "
    "adding four authors whose contribution to the re-executed study is "
    "described on the MDPI Change of Authorship Form submitted with these "
    "files. The CRediT statement in the manuscript matches that form.",
]

# --------------------------------------------------------------------------
# Reviewer 2
# --------------------------------------------------------------------------

R2_SUMMARY = [
    "Thank you very much for taking the time to review this manuscript. Every "
    "one of the fourteen points is addressed below, and four of them changed "
    "the study rather than the text.",

    "The comment on the Illicit Neighbour Ratio asked us to confirm that only "
    "training-set labels are used. We cannot: the feature reads neighbouring "
    "labels at every split, and the companion feature is worse. Both are "
    "withdrawn. The related comment on DGI inputs is answered by the same "
    "withdrawal, with a permutation test now added to prove the replacement "
    "features are label-independent.",

    "The request for computational cost, the request for per-time-step "
    "temporal results and the request for explainability have all been met "
    "with new measurements rather than with discussion, and the second of "
    "them produced one of the study's main findings: performance collapses at "
    "time step 43, and the round-one test window sits almost entirely inside "
    "the collapsed regime.",

    "On the Taproot comment we have had to reach the opposite conclusion from "
    "the one requested, and Response 2 gives the measurement that obliges us "
    "to. On the suggested reference in Response 5 we respectfully do not "
    "include it, and we give our reason rather than passing over the point.",

    "The study was re-executed from the raw data rather than edited around "
    "the comments. Eighteen experiments were run, each tied to a script, an "
    "output file and a dated commit in research/EXPERIMENT_LOG.md at " + REPO +
    "; the verification data are deposited at " + ZENODO + ". All revisions "
    "are marked in Manuscript_Revised_Highlighted.docx.",
]

R2_GENERAL = [
    ("Yes",
     "Thank you. The title has nonetheless been changed, because Reviewers 1 "
     "and 3 asked for the two tasks to be named separately. It is now "
     "“" + TITLE + "”."),
    ("No",
     "Addressed in Response 5. The literature review has been extended where "
     "the revision changed the claims, in particular on leakage-free "
     "evaluation of graph models and on protocol-level CoinJoin "
     "identification. The specific article the reviewer suggests is the one "
     "point in this letter where we respectfully differ, and Response 5 gives "
     "the reason."),
    ("No",
     "Addressed in Responses 3, 4, 6, 7 and 11. The label-derived features "
     "are withdrawn, the DGI input is now the 165 distributed features alone, "
     "the cluster-to-class mapping is fitted on training data and frozen, the "
     "screening label is stated as the exact rule it is, and the "
     "computational protocol is reported in full."),
    ("No",
     "Addressed in Responses 8, 9, 10, 13 and 14. Figure 4 of round one is "
     "corrected, the seed count is raised from five to twenty, the "
     "validation-selected threshold is marked on the threshold figure, "
     "per-time-step results are added, and the conclusion no longer describes "
     "self-supervised pre-training as the primary performance driver."),
    ("No",
     "Addressed in Response 5 for the suggested article. Separately, the "
     "reference cited for DBSCAN was mismatched, as Reviewers 1 and 3 both "
     "noted, and is corrected to Ester, Kriegel, Sander and Xu (1996). Every "
     "remaining citation was then checked against the claim it is used to "
     "support."),
    ("Yes",
     "Thank you. The contribution is now stated more narrowly than in round "
     "one, and the revision adds a validation that did not exist before: "
     "4,491 transactions in the corpus are confirmed as CoinJoin outputs "
     "against published protocol rules using blockchain data."),
    ("Yes",
     "Thank you. The two specific figures the reviewer raised have "
     "nonetheless been corrected, in Responses 8 and 10, and every figure "
     "drawn from the superseded ablation protocol has been regenerated."),
]

R2_POINTS = [
    (
        "The manuscript contains informal expressions such as “sort "
        "of,” “kind of,” and “so yeah.” The authors "
        "must revise the language to maintain an academic writing style.",
        [
            "Thank you for pointing this out. We agree, and a full editorial "
            "pass has been made over the whole manuscript. The three "
            "expressions named, and the other conversational constructions "
            "they were symptomatic of, are removed.",

            "In several of those places the informality was carrying a claim "
            "that the evidence did not support, so the sentences were not "
            "merely restyled but restated at the strength the results allow.",
        ],
    ),
    (
        "In Lines 76–78, the structural clustering is tested only on "
        "P2WPKH transactions as a Taproot proxy. Claims related to Taproot "
        "should therefore be moderated until validated using actual P2TR "
        "data.",
        [
            "Thank you. The proxy is no longer needed, and we have to report "
            "a conclusion opposite to the one requested, so we set out the "
            "evidence rather than simply asserting it.",

            "The claim that the corpus contained no confirmed P2TR "
            "transactions came from six boolean script-type columns, which "
            "are identically False for every one of the 5,884,387 rows. They "
            "report nothing at all about any script type. The round-one "
            "manuscript read a broken column as a finding, and the error was "
            "ours.",

            "Parsing the node-reported script descriptors instead gives the "
            "composition in Table 3: P2TR is 24.10 percent of outputs, "
            "2,494,168 transactions or 42.39 percent of the corpus involve "
            "Taproot, and 71.94 percent of CoinJoin-like candidates do. The "
            "corpus runs from 2022-07-13 to 2025-07-01 and is post-Taproot "
            "throughout.",

            "The structural clustering is therefore now evaluated on actual "
            "P2TR data, which is what the comment asks for as the condition "
            "for not moderating the claims. Experiment E07; the parsing "
            "script and its output are in the repository.",
        ],
    ),
    (
        "The Illicit neighbor Ratio uses labels of neighbouring transactions. "
        "The authors must clearly confirm that only training-set labels are "
        "used when computing this feature.",
        [
            "We cannot confirm it, and that is the honest answer. The feature "
            "reads neighbouring labels at every split, and the companion "
            "feature is worse: shortest_path_to_illicit is a breadth-first "
            "distance from every confirmed illicit node, and each illicit "
            "node is its own source, so the feature is exactly zero for all "
            "4,545 illicit nodes and non-zero for all 42,019 licit ones. The "
            "rule “shortest path equals zero implies illicit” has "
            "precision 1.000 on the test split.",

            "Supplying it drives the illicit-class F1, PR-AUC and Matthews "
            "correlation to 1.0000 under two unrelated model families, "
            "against 0.62 and 0.61 for the 165 distributed features alone. "
            "Table 1 reports the full four-way ablation.",

            "No leakage-free variant can be built either. The Elliptic edge "
            "set contains no cross-time-step edges at all, 0 of 234,355, so "
            "the graph is a disjoint union of forty-nine per-step snapshots "
            "and a strictly constructed version of both features is constant "
            "on validation and test.",

            "Both features are withdrawn from the manuscript, the abstract "
            "and the contribution statement. Experiments E01 and E02.",
        ],
    ),
    (
        "DGI is described as unsupervised, but its input contains engineered "
        "features that may depend on known illicit labels. The authors should "
        "clarify this point and evaluate DGI without label-derived features.",
        [
            "Agree, and the reviewer is right about the input. The point is "
            "now moot in the direction the comment asks for: the label-"
            "derived features are withdrawn, so DGI is fitted on the 165 "
            "distributed Elliptic features alone, with no engineered input at "
            "all. Every DGI result reported in the revision is from that "
            "configuration.",

            "We also added a check rather than relying on the argument. The "
            "label-free topological descriptors developed as candidate "
            "replacements are verified by permuting every label in the "
            "dataset and confirming the computed features are bit-identical; "
            "the test is in research/src/features.py and runs as part of the "
            "pipeline. Those replacements did not recover the lost "
            "performance, and that negative result is reported rather than "
            "omitted.",
        ],
    ),
    (
        "The following article cybersecurity weakness of the study Capacity "
        "and Spectrum-Aware Communication Framework for Wireless Sensor "
        "Network-based Smart Grid Applications can be discussed in the "
        "literature review section.",
        [
            "Thank you for the suggestion. We have considered it and "
            "respectfully do not include it, and we would rather give the "
            "reason than add the citation without one.",

            "The article concerns capacity-aware and spectrum-aware "
            "communication for wireless sensor networks in smart grid "
            "deployments. The present study concerns leakage in graph-based "
            "classifiers and protocol-level identification of Bitcoin "
            "CoinJoin transactions. We could not identify a claim in the "
            "revised manuscript that the article would support, and a "
            "citation that does not support a claim does not serve the "
            "reader.",

            "If the reviewer or the Academic Editor sees a specific "
            "connection we have missed, we would be glad to be told what it "
            "is and will add the reference against that claim.",
        ],
    ),
    (
        "In Section 3.4, cluster CoinJoin labels are assigned using majority "
        "voting over ground-truth cluster members. A separate training and "
        "evaluation procedure is needed to avoid possible label leakage.",
        [
            "Agree, and corrected. Clusters are now fitted on training-period "
            "transactions only, the cluster-to-class mapping is frozen, and "
            "test transactions are assigned by nearest fitted centroid "
            "without contributing to the mapping.",

            "Two further changes were needed that the comment did not reach. "
            "Round one had no temporal separation on this corpus at all, so a "
            "chronological split was introduced, with training to 2024-06, "
            "validation to 2024-12 and test on 2025. The majority vote itself "
            "was replaced by a positive-rate threshold selected on the "
            "validation split, because majority vote is ill-defined for the "
            "small clusters this corpus produces.",

            "Results under the corrected protocol are in Table 9. Experiment "
            "E08.",
        ],
    ),
    (
        "The author-curated dataset uses an is_coinjoin_like label derived "
        "from structural heuristics. The authors should avoid calling this "
        "strict ground truth and discuss possible circularity between the "
        "labels and structural features.",
        [
            "Agree. The term is removed throughout, the rule is now stated "
            "exactly, and the circularity is quantified rather than "
            "discussed.",

            "The rule. A depth-three decision tree reproduces the field with "
            "accuracy 1.00000000, and the identity is_coinjoin_like == "
            "(input_count >= 4) AND (output_count >= 4) holds for every one "
            "of the 5,884,387 transactions in the corpus.",

            "The circularity cannot be removed by dropping columns. Both "
            "counts were among the features the round-one detector clustered "
            "on, and they are recoverable by arithmetic from the remaining "
            "columns along two routes: transaction size is an affine function "
            "of the counts, and avg_input_value is total_input_value divided "
            "by input_count, so the quotient recovers input_count exactly for "
            "99.96 percent of rows and reproduces the label for 100.0000 "
            "percent of sampled transactions. The manuscript therefore "
            "reports a gradient across progressively stricter feature "
            "regimes, in Table 2, rather than a single score.",

            "Because no rearrangement of the corpus columns can settle "
            "whether the heuristic detects anything, the question was taken "
            "outside the corpus: 6,209 candidates were tested against "
            "published protocol rules using per-output values from public "
            "block explorers, and 4,491 are confirmed as CoinJoin outputs "
            "(Table 10). Against those confirmations the screening label "
            "attains a precision of 4.07 percent at near-total recall (Table "
            "11). Experiments E06, E14 and E14b.",
        ],
    ),
    (
        "Figure 4 is inconsistent. The pie chart includes unknown nodes, "
        "while the caption refers to the distribution of labelled nodes. The "
        "figure and caption should be corrected.",
        [
            "Agree, and corrected. The figure now plots the class "
            "distribution across labelled nodes only, with unknown nodes "
            "excluded, which is what the caption describes. The caption also "
            "now states the resulting 9.2 to 1 licit-to-illicit imbalance, "
            "since that ratio is what motivates the class-weighted loss used "
            "later.",

            "It is Figure 3 in the revised manuscript, the numbering having "
            "shifted with the other figure changes.",
        ],
    ),
    (
        "Table 1 shows substantial variation for GAT without DGI. More random "
        "seeds would provide stronger evidence regarding model stability.",
        [
            "Agree. The seed count is increased from five to twenty "
            "throughout, and the no-DGI arm is also retrained correctly, with "
            "the fused slot zeroed from the first epoch rather than ablated "
            "at inference, which was a separate defect Reviewer 1 identified.",

            "The stability claim survives both changes: over twenty seeds the "
            "cross-seed standard deviation falls from 0.0594 for the "
            "correctly retrained no-DGI arm to 0.0303 for the fused model, "
            "approximately a halving (Table 15).",

            "Because seed variance is not the same quantity as generalisation "
            "variance, a test-set bootstrap was added as well: 5,000 paired "
            "resamples of the 8,841 labelled test nodes, with both systems "
            "scored on the same resampled nodes (Table 16).",
        ],
    ),
    (
        "Figure 11 should clearly indicate the validation-selected threshold "
        "and the corresponding independent test performance.",
        [
            "Agree with the requirement, and it is met, though not on that "
            "figure. The round-one Figure 11 does not survive the revision: "
            "every figure was regenerated after Reviewer 1 found that several "
            "came from a superseded ablation protocol, and the threshold "
            "sweep for the Elliptic classifier is no longer among them.",

            "Both quantities the comment asks for are now reported "
            "explicitly, and together. The threshold is validation-selected "
            "and frozen for every run; for the primary seed of the fused "
            "model it is 0.8292, and Table 6 profiles the independent test "
            "split at exactly that operating point, giving the count, mean "
            "time step, mean degree and mean score of each of the four "
            "outcome groups. Threshold-free metrics, PR-AUC and ROC-AUC, are "
            "reported alongside every thresholded figure in the manuscript, "
            "so that no claim rests on the choice of operating point alone.",

            "The underlying protocol was also made consistent, since Reviewer "
            "1 found that the manuscript described two different threshold "
            "rules in different places. The threshold now maximises "
            "validation F1, is selected independently for each run, and is "
            "frozen before the test split is scored; research/src/metrics.py "
            "is the only code path that produces one.",

            "If the reviewer would prefer the operating point marked "
            "graphically as well, we will add it to the precision-recall and "
            "ROC figure.",
        ],
    ),
    (
        "The authors should report training time, inference time, GPU or CPU "
        "specifications, peak memory usage, and computational complexity for "
        "practical deployment.",
        [
            "Agree, and all five are now reported, in Table 5. The "
            "measurements were taken with a single process on an otherwise "
            "idle device, and the benchmark refuses to start if another CUDA "
            "process is present, because an earlier run was contaminated by "
            "exactly that.",

            "Hardware and software: NVIDIA RTX 4060 Laptop GPU with 8 GB "
            "VRAM and 24 multiprocessors, Intel Core i9 13th generation, 64 "
            "GB RAM, Windows 11, Python 3.10.10, PyTorch 2.7.1 with CUDA "
            "12.8, PyTorch Geometric 2.7.0.",

            "Training time and peak memory, as Table 5 reports them: DGI "
            "pre-training 25.36 s and 1,688.5 MB transductive, 15.05 s and "
            "1,312.3 MB inductive; fusion 73.13 s and 6,296.4 MB; GAT with "
            "the DGI slot zeroed 15.46 s and 6,298.0 MB; GAT 38.41 s and "
            "6,298.3 MB; GCN 9.99 s and 962.2 MB; GraphSAGE 8.12 s and "
            "946.8 MB. Inference is a full-graph forward pass over all "
            "203,769 nodes, averaged over twenty repetitions after warm-up: "
            "78.67 ms for fusion, 75.80 ms for GAT, 20.13 ms for GraphSAGE "
            "and 14.14 ms for GCN. Per-layer complexity is stated in the "
            "manuscript.",

            "Two points are worth drawing out for deployment. Attention "
            "dominates memory, roughly 6.3 GB against under 1 GB for the "
            "other encoders, because per-edge coefficients must be "
            "materialised across 468,710 directed edges. And the cheapest "
            "graph model is also the most accurate, which is part of why the "
            "round-one conclusion about pre-training has been withdrawn.",
        ],
    ),
    (
        "Section 5 the paper discusses operational forensic use, but model "
        "explainability is limited. An analysis using feature attribution or "
        "a suitable GNN explanation method would strengthen practical "
        "applicability. The authors can get some novel ideas from the study "
        "https://doi.org/10.1002/eng2.12963",
        [
            "Agree, and an attribution and failure analysis has been added as "
            "a section in its own right rather than as discussion.",

            "It reports which features the tabular and graph models actually "
            "use, and profiles the four outcome groups at the "
            "validation-selected operating point of 0.8292 for the primary "
            "seed, by mean time step, mean degree and mean score (Table 6). "
            "The result that matters operationally is that false positives "
            "concentrate on well-connected transactions, and that the graph "
            "model spends part of its attribution on neighbourhood statistics "
            "that have shifted by the test period. Both are carried into the "
            "forensic-practice discussion.",

            "We are grateful for the pointer to the suggested study. We have "
            "not cited it, because we were not able to establish a "
            "methodological link to the attribution analysis actually used "
            "here, and we would rather not cite a work we cannot tie to a "
            "specific claim. If the reviewer considers the connection a close "
            "one, we will gladly add it against the relevant passage.",
        ],
    ),
    (
        "Lines 788–820 The limitations are useful, but temporal "
        "performance should also be reported separately for individual "
        "Elliptic test time steps to evaluate concept drift.",
        [
            "Agree, and this request produced one of the main findings of the "
            "revision, so we report it at length rather than as an addition "
            "to the limitations.",

            "Broken out by time step, the classifier achieves a mean "
            "illicit-class F1 of 0.8582 across steps 35 to 42 and 0.0283 "
            "across steps 43 to 49, with recall falling from 0.789 to 0.000 "
            "at step 43, the dark-market shutdown. Figure 8 and Table 8 give "
            "the breakdown.",

            "Two consequences follow. The round-one test window lies almost "
            "entirely inside the collapsed regime, so the aggregate figure "
            "reported there was concealing a regime change rather than "
            "summarising a stable one; Table 7 compares the two evaluation "
            "windows directly. And it explains the appeal of the leaked "
            "feature discussed in Response 3: after the shock nothing learned "
            "from the training period still works, so a feature that "
            "re-encodes the label is the only thing holding the metric up.",

            "The manuscript now states that performance is not stationary and "
            "that deployment must assume so, and “Aggregate metrics that "
            "conceal regime change” is listed as a threat to validity in "
            "graph-based forensic evaluation generally.",
        ],
    ),
    (
        "The conclusion overstates some findings by describing DGI as the "
        "primary performance driver based mainly on one benchmark. This "
        "conclusion should be limited to the evaluated datasets and "
        "experimental conditions.",
        [
            "Agree, and the claim is not merely scoped but withdrawn, because "
            "the corrected experiments do not support it even on the one "
            "benchmark.",

            "Across twenty seeds a plain GraphSAGE baseline reaches F1 0.5467 "
            "against 0.5197 for the fused model, using 29,570 parameters "
            "instead of 131,588, one ninth of the training time and one sixth "
            "of the peak memory. Every graph model trails a Random Forest on "
            "the same features at 0.6212.",

            "What the evidence does support is narrower, and is what the "
            "manuscript now says: the frozen self-supervised representation "
            "improves the graph attention arm specifically, in both mean and "
            "variance, and that arm is not the strongest model examined. The "
            "claim is stated for the Elliptic Data Set under the protocol "
            "described, and the conclusion no longer generalises beyond it.",
        ],
    ),
]

R2_ENGLISH = (
    "The review report selects “The English could be improved to more "
    "clearly express the research”, and the first detailed comment "
    "identifies the informal expressions “sort of”, “kind "
    "of” and “so yeah”.",
    [
        "Thank you. A full editorial pass has been made over the whole "
        "manuscript and the three expressions named are removed, together "
        "with the other conversational constructions they were symptomatic "
        "of. The corrupted sentence in the round-one Section 5.1, raised by "
        "Reviewer 3, is corrected.",

        "The larger part of the problem was not style. Several passages "
        "claimed more than the evidence supported, and the clarity improves "
        "chiefly because those claims are now stated at the strength the "
        "results allow.",
    ],
)

R2_EXTRA = [
    "Where to find things. The clean file is Manuscript_Revised_Clean.docx "
    "and the marked copy is Manuscript_Revised_Highlighted.docx, in which 267 "
    "of 309 substantive paragraphs are highlighted. The proportion is high "
    "because the study was re-executed rather than edited.",

    "Reproducibility. Every experiment has a script in research/experiments/, "
    "an output file in research/results/, and a dated commit recorded in "
    "research/EXPERIMENT_LOG.md at " + REPO + ". The verification data, the "
    "raw block-explorer responses, the collection manifest and the checksum "
    "list are deposited at " + ZENODO + ".",

    "The two points where we differ. We do not include the suggested "
    "smart-grid communication reference, for the reason given in Response 5, "
    "and we have not cited the suggested explainability study, for the reason "
    "given in Response 12. Both remain open to the Academic Editor's "
    "direction; neither affects a result.",

    "Authorship. A change of authorship request accompanies this revision, "
    "adding four authors whose contribution to the re-executed study is "
    "described on the MDPI Change of Authorship Form submitted with these "
    "files. The CRediT statement in the manuscript matches that form.",
]

# --------------------------------------------------------------------------
# Reviewer 3
# --------------------------------------------------------------------------

R3_SUMMARY = [
    "Thank you very much for taking the time to review this manuscript, and "
    "for a report that was unusually precise about where the problems were. "
    "All ten points are addressed below.",

    "The circularity comment in item 2 is the one that reshaped the study. "
    "The reviewer asked us to justify or independently validate the labels. "
    "We did the second, because the first turned out to be impossible: the "
    "label is an exact two-variable threshold rule, and its defining "
    "variables are recoverable by arithmetic from the remaining columns, so "
    "no choice of features can make the evaluation non-circular. The question "
    "was therefore taken outside the corpus entirely, and 4,491 transactions "
    "are now confirmed as CoinJoin outputs against published protocol rules "
    "using blockchain data.",

    "On item 3 we have had to reach the opposite conclusion from the one "
    "requested. The absence of confirmed Taproot data was an artefact of six "
    "corpus columns that are identically False for every row; the corpus is "
    "post-Taproot throughout and is 24.10 percent P2TR by output. Response 3 "
    "gives the measurement.",

    "On item 10 we must also correct the reviewer, in the direction least "
    "favourable to the study: the two graph-structural features listed as a "
    "genuine strength are a label leak and are withdrawn, and the broader "
    "conclusion about self-supervised pre-training does not survive either. "
    "Response 10 sets both out.",

    "The study was re-executed from the raw data rather than edited around "
    "the comments. Eighteen experiments were run, each tied to a script, an "
    "output file and a dated commit in research/EXPERIMENT_LOG.md at " + REPO +
    "; the verification data are deposited at " + ZENODO + ". All revisions "
    "are marked in Manuscript_Revised_Highlighted.docx.",
]

R3_GENERAL = [
    ("No",
     "Addressed in Response 1. The title is now “" + TITLE + "”. It "
     "names the two tasks separately, and the abstract and contribution "
     "statement were rewritten to present the modules as separately evaluated "
     "on separate datasets with separate labels."),
    ("Yes",
     "Thank you. No change was requested on this point. The introduction was "
     "edited only where the revision required it, chiefly to drop the "
     "end-to-end pipeline framing and to follow the corrected Taproot "
     "measurement."),
    ("No",
     "Addressed in Responses 2, 3, 4 and 5. The circularity is quantified and "
     "then answered with external verification; the Taproot premise is "
     "corrected by direct measurement; the statistical basis is raised to "
     "twenty seeds plus a test-set bootstrap; and the heavy-tailed "
     "quantities, including the Mixing Index, are transformed with the "
     "transformation stated. The separate evaluation of the three modules is "
     "now stated as a limitation rather than presented as an integrated "
     "framework."),
    ("No",
     "Addressed in Responses 6 and 7, and in Response 10. The corrupted "
     "sentence is corrected; the single-seed figures are replaced by "
     "seed-aggregated versions or labelled explicitly as diagnostics; and two "
     "round-one conclusions are withdrawn because the corrected experiments "
     "do not support them."),
    ("No",
     "Addressed in Response 8. The reviewer is right that reference [27] was "
     "the Modified ROCK algorithm for categorical data, which is not DBSCAN. "
     "It is corrected to Ester, Kriegel, Sander and Xu (1996). Every "
     "remaining citation was then checked against the claim it is used to "
     "support."),
    ("Yes",
     "Thank you. The contribution is now stated more narrowly than in round "
     "one, and the revision adds the external validation the reviewer asked "
     "for in item 2, which did not exist before."),
    ("No",
     "Addressed in Response 7. The key figures are now seed-aggregated with "
     "variance bands, the remaining single-run figures are labelled as "
     "diagnostics, and every figure drawn from the superseded ablation "
     "protocol has been regenerated."),
]

R3_POINTS = [
    (
        "Title/scope mismatch. The title claims “GAT-Based Forensic "
        "Detection of Illicit CoinJoin Transactions,” but the GAT/DGI "
        "pipeline is only evaluated on Elliptic for generic illicit-node "
        "classification (no CoinJoin label exists there). Actual CoinJoin "
        "detection is performed by a separate DBSCAN module on a different, "
        "author-curated dataset. Please retitle or reframe to reflect the "
        "true three-module, two-dataset structure.",
        [
            "Agree. The title is now “" + TITLE + "”.",

            "It names the two tasks separately, keeps the system name for "
            "continuity with the submitted record, and no longer implies that "
            "the graph attention model detects CoinJoin transactions. The "
            "manuscript is reframed to match: the abstract, the contribution "
            "statement, Figure 1 and the conclusion describe three "
            "independently evaluated modules on two datasets, and the fact "
            "that no single dataset carries all three label types is now one "
            "of the entries in the Limitations section.",
        ],
    ),
    (
        "Circularity in ground truth. The author-curated corpus's label is "
        "itself derived from structural heuristics, and the proposed DBSCAN "
        "detector also clusters on structural features. This risks measuring "
        "agreement between two heuristic procedures rather than validated "
        "ground truth. Please justify or independently validate the labels "
        "(e.g., against known mixing-service address clusters).",
        [
            "Agree, and confirmed mechanically. It proved deeper than column "
            "selection can reach, so we took the second of the two options "
            "the reviewer offers.",

            "The label is an exact rule. A depth-three decision tree "
            "reproduces is_coinjoin_like with accuracy 1.00000000, and the "
            "identity is_coinjoin_like == (input_count >= 4) AND "
            "(output_count >= 4) holds for every one of the 5,884,387 "
            "transactions in the corpus. Both counts were among the features "
            "the round-one detector clustered on.",

            "Removing those columns does not repair the evaluation. The "
            "counts are recoverable by arithmetic from the remaining ones "
            "along two routes: transaction size is an affine function of the "
            "counts, and regressing the counts on the size family recovers "
            "the Bitcoin serialisation constants directly (vsize "
            "approximately 20.5 + 90.75 n_in + 35.15 n_out, R-squared 0.817); "
            "and avg_input_value is total_input_value divided by input_count, "
            "so the quotient recovers input_count exactly for 99.96 percent "
            "of rows and the reconstructed rule agrees with the distributed "
            "label for 100.0000 percent of sampled transactions. The "
            "manuscript therefore reports a gradient across progressively "
            "stricter feature regimes, in Table 2, rather than claiming any "
            "single regime is clean.",

            "Independent validation. Detection rules published with "
            "Dumplings, the reference tool used in the measurement literature "
            "to extract Wasabi, Whirlpool and JoinMarket transactions from "
            "mainnet, were applied to per-output values retrieved from public "
            "block explorers. All 6,209 protocol-shaped candidates were "
            "retrieved and tested, with no failures: 3,839 of 4,137 "
            "Whirlpool-shaped candidates confirm at 92.8 percent, 652 of "
            "1,522 WabiSabi-shaped candidates at 42.8 percent, and neither a "
            "random sample of 400 labelled positives nor one of 150 labelled "
            "negatives produced a single confirmation (Table 10). The 3,839 "
            "confirmed Whirlpool rounds distribute across the four "
            "denomination pools as 1,753, 1,103, 780 and 203, matching "
            "reported usage in which the smallest pool is busiest, which is a "
            "consistency check the procedure was not designed to pass. "
            "Against those confirmations the screening label attains a "
            "precision of 4.07 percent at near-total recall (Table 11).",

            "The confirmations also support a detection experiment whose "
            "labels do not originate in the heuristic under evaluation, which "
            "is the first such experiment in the study. It is deliberately "
            "guarded. Evaluation runs inside each candidate group, because "
            "pooling groups with positive rates of 92.8 and 42.8 percent "
            "would let a model score well by recognising the group and "
            "predicting its base rate. And a strict regime withholds every "
            "value and count column, because the mean output value of a "
            "confirmed Whirlpool round is its common output value. With only "
            "fee, size and timing remaining, the strongest single rule "
            "reaches F1 0.9663 and Matthews correlation 0.5823, and a trained "
            "model reaches 0.9957 and 0.9559; the margin lies in rejecting "
            "look-alikes that match a pool denomination exactly yet fail "
            "other protocol conditions. Two qualifications are stated in the "
            "manuscript: the result reflects the rigidity of a protocol that "
            "produces near-identical transactions, and it does not extend to "
            "implementations the verification cannot label, for which the "
            "weaker WabiSabi figures, recall 0.7377 at precision 1.000, are "
            "the better guide. Experiments E06, E14, E14b and E15.",
        ],
    ),
    (
        "Pre-Taproot limitation understated relative to claims. Given Taproot "
        "has been live since November 2021, using zero confirmed P2TR "
        "transactions throughout is a major limitation for a 2026 forensic "
        "tool. The abstract/conclusion should state this limitation as "
        "prominently as the performance numbers, not just in the discussion.",
        [
            "Thank you. We have to answer this one differently, because the "
            "premise does not hold, and we would rather set out the "
            "measurement than quietly drop the point.",

            "The claim that the corpus contained zero confirmed P2TR "
            "transactions came from six boolean script-type columns. Those "
            "six columns are identically False for every one of the 5,884,387 "
            "rows. They do not report the absence of Taproot; they report "
            "nothing at all about any script type, and the round-one "
            "manuscript read a broken column as a finding.",

            "Parsing the node-reported script descriptors instead gives the "
            "composition in Table 3. P2TR accounts for 24.10 percent of "
            "outputs; 2,494,168 transactions, or 42.39 percent of the corpus, "
            "involve Taproot; and 71.94 percent of CoinJoin-like candidates "
            "do. The corpus runs from 2022-07-13 to 2025-07-01 and is "
            "post-Taproot throughout.",

            "The limitation is therefore removed rather than made more "
            "prominent, and the proxy analysis is replaced by direct "
            "measurement on real P2TR data. Experiment E07; the parsing "
            "script and its output are in the repository.",

            "The address-count columns were later found to be broken in the "
            "same way, and the pattern is now discussed as a threat to "
            "validity in its own right, under “A derived column whose "
            "name describes a quantity it does not hold”, because it is "
            "not detectable from summary statistics and it affected two "
            "independent claims in round one.",
        ],
    ),
    (
        "Statistical basis is thin. Five seeds (df = 4) support the paired "
        "t-tests in Table 4, but the language in Sections 4.7 and 6 "
        "(“significant... so yeah you can treat it as meaningful”) "
        "overstates the strength of evidence this affords. Consider tempering "
        "claims or adding more seeds.",
        [
            "Agree, and both: seeds are increased from five to twenty, and "
            "the claims are tempered independently of the seed count.",

            "A test-set bootstrap was added as well, because seed variance "
            "and generalisation variance are not the same quantity and "
            "neither substitutes for the other. Table 16 reports percentile "
            "intervals from 5,000 paired resamples of the 8,841 labelled test "
            "nodes, with both systems scored on the same resampled nodes; "
            "Table 15 reports the seed-level paired t-tests over twenty "
            "seeds, with nineteen degrees of freedom, and is described only "
            "as evidence of seed-level consistency.",

            "The two analyses disagree in one place, and the manuscript says "
            "so rather than reporting the more convenient one. The bootstrap "
            "places the fused model above GraphSAGE by 0.0430 with an "
            "interval excluding zero; the seed-level test places GraphSAGE "
            "above the fused model by 0.0270. The subsection “The Two "
            "Analyses Disagree, and That Is Informative” sets out why "
            "both can be correct and what each is entitled to claim.",

            "The quoted phrasing is gone, along with the rest of the informal "
            "register, in the editorial pass described under Section 4 below.",
        ],
    ),
    (
        "Data inconsistency. The Mixing Index statistic (mean 8.79, SD "
        "689.27) implies severe outliers/skew that is never discussed, "
        "transformed, or bounded. Please address this or clarify how outliers "
        "affect downstream clustering.",
        [
            "Thank you. The reviewer was right that the statistic could not "
            "be what it was reported to be, and investigating the query "
            "uncovered a defect rather than a heavy tail.",

            "What was wrong. The corpus supplies columns named "
            "input_address_count, output_address_count and total_addresses. "
            "They do not hold the quantities those names describe. Each "
            "counts the elements of a one-element array containing a "
            "semicolon-joined address string rather than the distinct "
            "addresses inside it, so they sit at 1, 1 and 2 for almost every "
            "transaction and agree with the parsed address lists for only "
            "54.34 percent of rows. Computed from them the Mixing Index is "
            "identically 0.5 and its correlation with the screening label is "
            "0.0031. As distributed, the statistic carries no information at "
            "all.",

            "The corrected statistic. Parsing the address strings directly "
            "gives a mean of 1.89 input and 2.39 output addresses, with "
            "maxima of 1,794 and 3,199. Recomputed on those values the Mixing "
            "Index has a mean of 0.678, a standard deviation of 1.402, a "
            "median of 0.5 and a maximum of 250.5, with a skewness of 69.2. "
            "Neither the corrected nor the distributed computation reproduces "
            "the figures the reviewer quotes, which appear to derive from a "
            "third procedure the present study was unable to reconstruct; "
            "the mean of 8.79 and the standard deviation of 689.27 are "
            "therefore withdrawn rather than explained.",

            "How the outliers are handled, which is the second half of the "
            "comment. The corrected statistic is genuinely heavy-tailed, with "
            "5,885 transactions above the 99.9th percentile of 9.41, so it is "
            "used under a transform throughout, stated in the manuscript "
            "under “Handling the tail”. The transform reduces "
            "skewness from 69.2 to 7.9 and, more to the point, makes the "
            "quantity useful for the clustering that consumes it: correlation "
            "with the screening label rises from 0.0031 for the distributed "
            "column to 0.274 for the corrected statistic and 0.541 under the "
            "transform. Winsorising at the 99.9th percentile is reported for "
            "comparison in the released results but is not used, since it "
            "discards precisely the high-multiplicity transactions that are "
            "of forensic interest.",

            "This is the third column family in this corpus found to "
            "misreport its own contents, after the six script-type indicators "
            "discussed in Response 3 and the degenerate "
            "value_concentration_ratio. The pattern, a derived column whose "
            "name describes a quantity it does not hold, is not detectable "
            "from the schema or from summary statistics, and is now raised as "
            "a threat to validity in its own right, with the remedy the study "
            "adopted: check each derived column against a recomputation from "
            "the raw field it summarises.",
        ],
    ),
    (
        "Corrupted text. Section 5.1 contains a garbled/incomplete sentence: "
        "“...across different ininitializationsafrom36 to 0.687 "
        "depending on s...” This must be corrected before publication.",
        [
            "Agree, and corrected. The sentence is restored and the "
            "surrounding passage rewritten, since the corruption had removed "
            "the figure the sentence was reporting.",

            "The whole manuscript was then proofread for the same class of "
            "defect, which arose from a text-merge error rather than from "
            "typing, so a single instance was unlikely to be the only one.",
        ],
    ),
    (
        "Figure/table consistency. Figures 5–9 and 12 use a single "
        "representative seed (42) while Tables 1 and 4 report five-seed "
        "aggregates; the manuscript repeatedly has to caveat this. This dual "
        "reporting is confusing and risks visually overstating performance. "
        "Recommend replacing single-run figures with seed-aggregated versions "
        "(mean curves with variance bands) or moving them to supplementary "
        "material.",
        [
            "Agree, and adopted. The key figures, the precision-recall curve, "
            "the ROC curve and the confusion matrix, are now seed-aggregated "
            "with variance bands over twenty seeds. The figures that remain "
            "single-run are training diagnostics, for which a single run is "
            "the meaningful object, and each is labelled as such in its "
            "caption and reported alongside, not in place of, the twenty-seed "
            "aggregate.",

            "A second inconsistency Reviewer 1 identified was corrected at "
            "the same time: several round-one figures came from a superseded "
            "inference-only ablation. Every figure has been regenerated under "
            "the final protocol, and none from the superseded run survives.",
        ],
    ),
    (
        "Reference mismatch. Reference [27] (Saruladha & Likhitha, "
        "“Modified ROCK (MROCK) algorithm for clustering categorical "
        "data”) is cited as the basis for the DBSCAN method, but ROCK is "
        "a distinct categorical-clustering algorithm, not DBSCAN. Please cite "
        "the correct DBSCAN source (e.g., Ester et al., 1996) or clarify the "
        "connection.",
        [
            "Agree, and corrected to Ester, Kriegel, Sander and Xu (1996), "
            "the original DBSCAN paper. There is no connection to clarify: "
            "the citation was simply wrong.",

            "Every remaining citation was then checked against the claim it "
            "is used to support, and the reference list was updated where the "
            "revision changed the claim rather than only where a reference "
            "was mismatched.",
        ],
    ),
    (
        "Writing quality. Several passages (Sections 3, 4, 5) use informal, "
        "conversational phrasing (“sort of,” “kind of,” "
        "“yeah”) inconsistent with journal style. A full editorial "
        "pass is recommended.",
        [
            "Agree, and a full editorial pass has been made over the whole "
            "manuscript. The named expressions and the constructions they "
            "were symptomatic of are removed.",

            "In several of those places the informality was carrying a claim "
            "the evidence did not support, as the reviewer notes in item 4. "
            "Those sentences were restated at the strength the results allow "
            "rather than merely restyled.",
        ],
    ),
    (
        "Positive contributions. The honest reporting of the null result for "
        "the Multi-Dimensional Chamfer Distance extension, the DGI+GAT "
        "ablation showing reduced cross-seed variance, and the two new "
        "graph-structural features (shortest-path-to-illicit, "
        "illicit-neighbor-ratio) are genuine strengths and should be "
        "retained/highlighted.",
        [
            "We are grateful for the encouragement, and we must correct two "
            "of the three items, in the direction least favourable to the "
            "study.",

            "The two graph-structural features are a label leak and are "
            "withdrawn. shortest_path_to_illicit is a breadth-first distance "
            "from every confirmed illicit node, and each illicit node is its "
            "own source, so the feature is exactly zero for all 4,545 illicit "
            "nodes and non-zero for all 42,019 licit ones; the rule "
            "“shortest path equals zero implies illicit” has "
            "precision 1.000 on the test split. Supplying it drives F1, "
            "PR-AUC and Matthews correlation to 1.0000 under two unrelated "
            "model families (Table 1). No leakage-free variant can exist "
            "either, because the Elliptic edge set contains no cross-time-"
            "step edges at all, 0 of 234,355, so a strictly constructed "
            "version is constant on validation and test. We would rather lose "
            "the contribution than keep a result we cannot defend.",

            "The ablation finding survives, but the conclusion it supported "
            "does not. Measured against a properly retrained no-DGI arm over "
            "twenty seeds, the cross-seed standard deviation does fall from "
            "0.0594 to 0.0303. But across the same twenty seeds a plain "
            "GraphSAGE baseline reaches F1 0.5467 against 0.5197 for the "
            "fused model, using 29,570 parameters instead of 131,588, one "
            "ninth of the training time and one sixth of the peak memory, and "
            "every graph model trails a Random Forest on the same features at "
            "0.6212. The claim that self-supervised pre-training is the "
            "primary performance driver is therefore withdrawn and replaced "
            "by the narrower one the evidence supports: it improves the graph "
            "attention arm specifically, in both mean and variance, and that "
            "arm is not the strongest model examined.",

            "The third item stands. The Chamfer-distance null result is "
            "retained, and the revision contains several further negative "
            "results reported in the same spirit, including the failure of "
            "the label-free topological features developed as replacements "
            "for the withdrawn ones.",
        ],
    ),
]

R3_ENGLISH = (
    "The review report selects “The English could be improved to more "
    "clearly express the research”, and item 9 of the detailed comments "
    "recommends a full editorial pass.",
    [
        "Thank you. A full editorial pass has been made over the whole "
        "manuscript. The informal expressions are removed, and the corrupted "
        "sentence identified in item 6 is corrected.",

        "The larger part of the problem was not style. Several passages "
        "claimed more than the evidence supported, as item 4 notes, and the "
        "clarity improves chiefly because those claims are now stated at the "
        "strength the results allow. The revised manuscript is approximately "
        "20,300 words against approximately 12,400 in round one, and the "
        "added material was written to the same standard and read back for "
        "concision.",
    ],
)

R3_EXTRA = [
    "Where to find things. The clean file is Manuscript_Revised_Clean.docx "
    "and the marked copy is Manuscript_Revised_Highlighted.docx, in which 267 "
    "of 309 substantive paragraphs are highlighted. The proportion is high "
    "because the study was re-executed rather than edited; the unmarked text "
    "is largely the related-work and ethics material.",

    "Reproducibility. Every experiment has a script in research/experiments/, "
    "an output file in research/results/, and a dated commit recorded in "
    "research/EXPERIMENT_LOG.md at " + REPO + ". The verification data, the "
    "raw block-explorer responses behind it, the collection manifest and the "
    "checksum list are deposited at " + ZENODO + ", and the Data Availability "
    "Statement points to both.",

    "What the revision costs the study. Two round-one conclusions are "
    "withdrawn, as set out in Response 10, and the headline illicit-detection "
    "figure falls accordingly. We have reported the corrected numbers rather "
    "than the round-one ones, and the manuscript is now an audit of "
    "graph-based forensic evaluation as much as a detection result, which is "
    "what the title reflects.",

    "Authorship. A change of authorship request accompanies this revision, "
    "adding four authors whose contribution to the re-executed study is "
    "described on the MDPI Change of Authorship Form submitted with these "
    "files. The CRediT statement in the manuscript matches that form.",
]

REVIEWERS = {
    1: dict(summary=R1_SUMMARY, general=R1_GENERAL, points=R1_POINTS,
            english=R1_ENGLISH, extra=R1_EXTRA, verified=True),
    2: dict(summary=R2_SUMMARY, general=R2_GENERAL, points=R2_POINTS,
            english=R2_ENGLISH, extra=R2_EXTRA, verified=False),
    3: dict(summary=R3_SUMMARY, general=R3_GENERAL, points=R3_POINTS,
            english=R3_ENGLISH, extra=R3_EXTRA, verified=False),
}
