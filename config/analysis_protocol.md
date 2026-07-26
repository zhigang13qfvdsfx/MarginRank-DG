# TransSurv-PDAC analysis protocol

Protocol version: 0.2.0  
Date frozen: 2026-07-22  
Primary objective: develop and test a cohort-robust, missingness-aware, single-sample survival prediction framework for PDAC transcriptomic data.

## Research integrity constraints

1. Previously analysed cohorts (TCGA-PAAD, CPTAC-PDAC, GSE62452, and GSE85916) are development domains. They cannot be described as untouched final validation for the new algorithm.
2. GSE28735 is audit-only because it overlaps the GSE62452 source series and cannot enter an independent meta-analysis with GSE62452.
3. Candidate final holdouts are selected by eligibility rules before outcome-level analysis. Once designated, they are evaluated exactly once after the algorithm, hyperparameters, preprocessing, feature set, missing-gene rules, horizons, and metrics are frozen.
4. No final-holdout outcome may be used for feature selection, hyperparameter tuning, model selection, cutoff selection, recalibration, or cohort exclusion.
5. All attempted model families, configurations, failures, and decisions are logged. Results are reported irrespective of direction.

## Development domains

- TCGA_PAAD: RNA-seq, n=148, 85 events.
- CPTAC_PDACC: RNA-seq, n=134, 76 events.
- GSE62452: microarray, n=65, 49 events.
- GSE85916: microarray, n=79, 57 events; FAM83A absent.

## Candidate final-holdout eligibility

A dataset is eligible if it contains primary PDAC tumour tissue, patient-level overall survival time and censoring status, an expression platform with adequate gene mapping, at least 40 analysable patients, at least 20 observed deaths, no known sample overlap with a development domain, and enough follow-up for at least one prespecified horizon. Cohorts with only binary short/long survival labels are stress tests, not time-to-event holdouts.

Candidate primary holdouts: GSE21501 and GSE57495. Candidate secondary stress-test/holdout: GSE79668, subject to endpoint and overlap audit. ICGC cohorts are excluded unless access, endpoint provenance, and overlap can be resolved without using their outcomes for model development.

## Primary algorithm

TransSurv combines:

1. outcome-free single-sample expression transformation;
2. training-fold-only feature screening that rewards meta-analytic effect strength, direction consistency, and low between-study score dispersion;
3. a shared sparse Cox predictor with penalised study-specific coefficient deviations;
4. a transportability penalty based on smooth worst-domain loss;
5. nested leave-one-study-out model selection;
6. source-only model bundling for independent prediction.

The original locked six-gene model is the legacy reference. A prespecified common-gene candidate space may be used for the new algorithm, but all feature selection must occur inside the development resampling loop.

Protocol amendment rationale: the version 0.1 simulation pilot and its fresh-seed confirmation improved mean discrimination by approximately 0.009 but failed the worst-domain/dispersion gate. Inspection showed that domain-specific prognostic nuisance features remained in the consensus predictor. Version 0.2 therefore adds the originally intended cross-domain stability screen as an explicit algorithmic component. The original simulation outputs remain retained; no threshold or simulation outcome definition was changed.

## Missing-gene module

MaskSurv uses stochastic feature masking and compatible submodel distillation. Its primary stress test removes one active gene, matching the observed FAM83A absence in GSE85916. Secondary tests remove 2-3 genes and add platform-like measurement noise.

## Lightweight reliability module

SurvGuard is audit-only unless it passes its gate. It combines gene coverage, source-support distance, bootstrap/model disagreement, and preprocessing disagreement. It may flag an out-of-domain prediction but may not change the locked risk score.

## Prespecified metrics

Primary: patient-level Harrell C-index and Uno C-index in held-out studies.  
Secondary: integrated/time-specific Brier score, cumulative/dynamic AUC at 12, 24, and 36 months when estimable, calibration-in-the-large, calibration slope, flexible calibration curves, hazard ratio per SD, and prediction heterogeneity.  
Reliability: risk-coverage curves and retained-versus-flagged error enrichment.  
Uncertainty: patient bootstrap with 1,000 resamples for final analyses and at least 300 during development.

## Evidence gates

### Gate 1 - implementation and simulation

Proceed if the algorithm is numerically stable, shows no leakage in unit tests, and under prespecified batch shift/missingness simulations either improves worst-domain discrimination by at least 0.02 or reduces performance dispersion by at least 20% without reducing mean C-index by more than 0.01.

### Gate 2 - leave-one-study-out development

Top-tier signal (green): relative to the strongest simple baseline, mean held-out Uno C improves by at least 0.02, the worst-domain C-index improves by at least 0.03, at least three of four domains improve, no domain worsens by more than 0.02, and mean Brier score improves by at least 5%.

Methodologically promising (amber): mean Uno C improves by at least 0.01 or performance dispersion falls by at least 20%, with no material average loss and no unexplained cohort-specific failure. Amber results justify further development but not a top-tier claim.

Stop/reframe (red): gains occur only in resubstitution/random patient splits, depend on one domain, or disappear against elastic-net/stratified-Cox baselines.

### Gate 3 - missingness robustness

Proceed if one-gene masking causes no more than 0.02 absolute C-index loss and MaskSurv reduces the loss from naive zero/mean imputation by at least 50%, while preserving at least 0.85 rank correlation with the complete-feature prediction in source-domain masking experiments.

### Gate 4 - reliability audit

Retain SurvGuard in the main paper only if the flagged group has materially larger prediction error and 80% retained coverage improves Brier score by at least 10% or C-index by at least 0.02 in cross-study validation. Otherwise report it as software QC only.

### Gate 5 - one-time final holdout

A strong final result requires concordant direction in both primary holdouts, mean Uno C at least 0.60, no primary holdout below 0.55 unless explained by a prespecified data-quality issue, and calibration slopes broadly compatible with 0.7-1.3 at supported horizons. These thresholds assess strength; failure is reported and cannot trigger holdout-specific retuning.

## Model-development boundaries

Permitted before final lock: transformation family, transportability penalty, regularisation strength, feature-space size, masking probability, and reliability threshold, all selected within nested leave-one-study-out development.  
Not permitted: external-outcome-driven normalization, post-hoc endpoint switching, optimized external cutoffs, excluding a cohort for an unfavourable result, or repeated testing of the final holdouts.

## Post-v1 amendment: calibration-focused v2 cycle

Amendment frozen: 2026-07-22, before access to individual outcomes in the v2 confirmation cohorts.

The v1 one-time validation remains final and immutable. It showed reproducible discrimination in GSE57495 and GSE79668 but failed the prespecified calibration component of Gate 5 because the GSE57495 calibration slope was 0.418. These results may motivate a new model-development cycle, but neither cohort may be used to select v2 features, hyperparameters, ensemble weights, calibration parameters, or reliability thresholds. All v1 results and completion locks are retained.

### v2 development domains

E-MTAB-6134 is added to TCGA_PAAD, CPTAC_PDACC, GSE62452, and GSE85916 as a fifth development domain. Its individual OS metadata were inspected before this amendment, so it is explicitly not an untouched external validation cohort. Only primary PDAC samples with valid OS time and event status enter model development.

### v2 model-development rules

1. The candidate model family remains the v1 dual-objective relative-expression framework: a top-three pooled survival-learner branch and a top-three IPCW pairwise-ranking branch.
2. The existing feature-space, learner, regularisation, robustness, and ensemble-weight grids are reused without adding outcome-responsive candidates. Candidate ranking uses mean held-out Uno C, worst-domain Uno C, and between-domain dispersion in five-domain leave-one-study-out validation.
3. Every feature screen and fitted coefficient is training-domain-only. Single-sample ranks remain the deployable input representation; cohort z scores are permitted only for training-fold feature screening.
4. A single global risk-temperature parameter is estimated solely from the five held-out development predictions. It is the median of cohort-specific univariable Cox calibration slopes, clipped to the prespecified interval 0.40-1.50. The same multiplier is then applied to all future predictions; it cannot change discrimination.
5. Breslow baseline hazards and the SurvGuard reliability layer are refitted from development data only. SurvGuard remains a warning/coverage layer and may not alter the risk score.
6. No individual v2 confirmation outcome may be opened until the model bundle, transforms, selected features, temperature, horizons, missing-gene rule, and analysis script are hashed and locked.

### v2 development gate

Proceed to fresh confirmation only if five-domain leave-one-study-out validation has mean Uno C at least 0.60, minimum Uno C at least 0.55, mean Harrell C no more than 0.01 below the v1 four-domain value, at least four of five temperature-scaled calibration slopes within 0.70-1.30, and no scaled slope outside 0.50-1.50. Failure triggers transparent re-evaluation within development data or reframing; it does not permit opening confirmation outcomes.

### Fresh v2 confirmation cohorts

- Primary: PACA-AU / QCMG 2016, using primary-tumour RNA-seq and patient-level OS.
- Co-primary: GSE71729, using primary-pancreas PDAC samples and patient-level OS.
- Conditional contemporary stress test: GSE205154, only if public sample-level OS time/status and primary-tumour identifiers can be recovered without access-controlled clinical data.

Eligibility remains at least 40 analysable patients, at least 20 deaths, adequate selected-pair coverage, and no detected sample overlap with development domains. A candidate failing these objective rules is excluded with the reason logged before prediction. GSE57495 and GSE79668 are not v2 confirmation cohorts and are reported only as the completed v1 validation.

### v2 confirmation gate

Top-tier support requires concordant risk direction in both co-primary cohorts, mean Uno C at least 0.60, neither Uno C below 0.55, calibration slopes broadly compatible with 0.70-1.30, and no unresolved critical proportional-hazards or overlap issue. The contemporary stress test is supportive rather than required. All confidence intervals, negative findings, and protocol deviations are reported; no result is guaranteed or cosmetically selected.

## Post-v2 development amendment: BatchScale v2.1

Amendment recorded: 2026-07-22, after the v2 five-domain development gate failed and before access to any individual v2 confirmation outcome.

The original v2 result is retained as a failed attempt: mean Uno C was 0.605 and minimum Uno C was 0.576, but a single raw-score temperature placed only three of five calibration slopes in 0.70-1.30. Examination of outcome-free score distributions showed substantial between-cohort variation in the dispersion of the held-out prognostic index. This is a risk-scale transport problem rather than a ranking failure.

BatchScale v2.1 therefore adds a cohort-mode calibration layer without changing feature selection, fitted components, ensemble weights, or within-cohort ranks:

1. For a deployment batch with at least 40 analysable samples, centre the frozen raw risk by its batch mean and divide by its batch population standard deviation. This operation uses no outcomes.
2. Estimate one global shrinkage coefficient from the complete five-domain out-of-fold predictions using a Cox model stratified by development cohort and ridge penalizer 0.01. The coefficient is then frozen and multiplies every standardized future batch score.
3. BatchScale may improve risk-scale transport but cannot improve Harrell C, Uno C, AUC, or any other rank statistic within a cohort. Its contribution is judged only by calibration, Brier score, and probability accuracy in fresh confirmation data.
4. For a single sample or a deployment batch smaller than 40, the software returns the unscaled relative-risk rank with a calibration warning; it does not estimate a batch scale from an unstable sample distribution.
5. The v2 development gate is re-evaluated once with BatchScale. If it passes, all model components and BatchScale parameters are frozen before accessing v2 confirmation outcomes. No alternative target dispersion or penalty is tested against confirmation outcomes.

This amendment is explicitly a development-data-driven algorithm revision, not a prespecified success. Its value must be established in the fresh co-primary confirmation cohorts.

## SurvGuard v2.1 enhancement audit

Amendment recorded: 2026-07-22, after the original five-domain SurvGuard audit failed Gate 4 and before v2 confirmation outcome access.

The failed reliability result is retained: at 80% coverage, mean Harrell C improved by 0.0129 rather than the required 0.02, and E-MTAB-6134 did not improve. One final development-only enhancement is permitted before deciding whether to downgrade SurvGuard to software QC.

The enhanced audit adds only outcome-free uncertainty features: disagreement among the three frozen Cox components, prediction variation when each source domain is omitted, relative-expression pair tie margin, local risk isolation, and the existing expression-shift distances and risk magnitude. Reliability models are restricted in advance to an unweighted uncertainty mean, ridge regression, shallow random forest, and depth-1/depth-2 gradient boosting. Model choice remains nested across source cohorts. The locked prognostic risk is never altered; SurvGuard only ranks warnings/coverage. Gate 4 and its 80% coverage threshold are unchanged. If this enhancement fails, SurvGuard is reported as QC only and no further reliability variants are promoted.

## Post-v2.1 amendment: CalibraGuard v3

Amendment locked: 2026-07-22, after the completed one-time PACA-AU/GSE71729 confirmation and before bulk access to outcomes in any fresh v3 confirmation cohort.

The v2.1 confirmation is immutable. It preserved direction and produced a mean Uno C of 0.607, but calibration slopes of 1.417 in PACA-AU and 0.464 in GSE71729 failed Gate 5. v3 therefore does not alter the frozen 17-gene relative-expression ranker or reinterpret the v2.1 result. Instead, it treats cross-cohort effect-scale heterogeneity as the primary methodological problem.

PACA-AU and GSE71729 now join the five original development domains and the two completed v1 holdouts solely for a new calibration-transport cycle. Leakage-resistant predictions are mandatory: nested out-of-cohort predictions are used for the original development cohorts, while only the already frozen predictions are used for former holdouts. No resubstitution metric may be reported as transport performance.

CalibraGuard predicts the positive Cox coefficient for a one-standard-deviation frozen raw risk from prespecified outcome-free cohort descriptors. Candidate families are restricted to a pooled coefficient, a shrinkage assay-class coefficient, ridge meta-regression, and a convex shrinkage blend. Selection and interval construction are fully nested leave-one-cohort-out. A finite-sample conformal interval quantifies coefficient uncertainty; absolute survival output is suppressed when this interval is not sufficiently informative. The complete candidate grid, gates, and forbidden actions are recorded in `V3_CALIBRAGUARD_PROTOCOL.lock`.

Fresh v3 confirmation candidates are locked in `V3_CONFIRMATION_CANDIDATES.lock`. GSE183795 is the primary microarray cohort. GSE224564 is eligible as an RNA-seq co-primary only after patient/source-cohort de-duplication against GSE183795, because the records may share German cases. GSE205154 remains a conditional FFPE stress test and requires publicly traceable individual OS. Individual risk predictions and risk-outcome analyses are prohibited until the v3 model, inference behavior, analysis script, and eligibility audit are hashed and frozen.

### CalibraGuard v3.1 stability amendment

The first v3 development gate failed and is retained in `results/26_v3_development_gate.json`. Its nested selector varied across outer folds, achieved only three of nine slopes in 0.70-1.30, covered seven of nine coefficients, and worsened median absolute log-slope error by 13.95% relative to BatchScale. This failure occurred before any bulk fresh-confirmation outcome access.

One dimension-reduced revision is permitted. It adds only model-internal, outcome-free stability descriptors—component concordance, component dispersion, selected-pair entropy, informative-pair fraction, branch correlation, and branch disagreement—and removes the unstable six-variable distributional meta-regression. Candidate feature sets, penalties, blends, global-selection rule, and required leave-two-domain/perturbation stability analyses are frozen in `V3_1_STABILITY_AMENDMENT.lock`. Gate thresholds do not change. If v3.1 fails, no further descriptor expansion is permitted; the package is reframed as relative-risk-only before any fresh outcome analysis.

### CalibraGuard v3.2 data-expansion amendment

v3.1 also failed and is retained in `results/28_v3_1_development_gate.json`. It came close to the locked boundary—14.88% rather than 15% median error improvement, five rather than six narrow-range slopes, eight of nine broad-range slopes, and eight of nine conformal coverage—but is not declared successful.

Before generating any risk prediction in GSE183795, that entire study is reclassified as a tenth development domain. It can no longer be called external validation. v3.2 reruns exactly the v3.1 candidate grid and descriptors; no new algorithmic degree of freedom is added. The ten-domain gate uses the same proportional requirements: at least seven narrow-range slopes, nine broad-range slopes, 80% interval coverage, 15% median error improvement, mean Uno C at least 0.60, minimum Uno C at least 0.55, and positive direction in every domain. Details are frozen in `V3_2_DATA_EXPANSION_AMENDMENT.lock`.

Fresh confirmation priority becomes non-overlapping GSE224564 RNA-seq, public-clinical GSE205154 FFPE RNA-seq, and conditional GSE277386 as a treated-population stress test. Patient/source de-duplication against GSE183795 is mandatory. No fresh risk-outcome analysis occurs before v3.2 is frozen.
## v3.3 relative-risk reframe after the calibration stop rule

The v3.0, v3.1, and v3.2 calibration-development gates all failed their
prespecified targets. Absolute point-survival calibration development is
therefore closed. The unchanged v2.1 ordering function may proceed only as the
v3.3 research-only relative-risk model described in
`V3_3_RELATIVE_RISK_REFRAME.lock`. Absolute survival probabilities, survival
times, clinical thresholds, and clinical-utility claims are prohibited.

Before any GSE224564 risk score is generated, the UMD-only eligibility files,
rank-only bundle, and one-time validation script must be hashed. The German
source cohorts are excluded because of overlap with GSE183795 development
samples. GSE205154 and GSE277386 may be logged as ineligible but may not be
silently substituted or pooled. Green, amber, and red result interpretations
are fixed in the v3.3 lock and cannot be altered after outcome association is
calculated.

## v4 redesign after the v3.3 external failure

The v3.3 UMD-2/UMD-3 confirmation is red and permanently unavailable for
tuning. Previously consumed v1/v2 validation cohorts may be migrated into v4
development only with their former roles disclosed. The ten primary-tumour
development domains and the separately labelled GSE205154 metastatic auxiliary
domain are specified in `V4_DEVELOPMENT_MIGRATION.lock`. Any v4 model must pass
fully nested leave-one-domain-out development gates before a new external score
is generated.

The PRINCE trial is reserved as a metastatic transportability stress test and
GSE77435 as a small binary extreme-survivor stress test. Neither repairs the
absence of a new public resected-primary time-to-event cohort, and neither may
be described otherwise. Absolute-survival prediction remains disabled.

### Pre-score PACA-CA candidate amendment

The PRINCE eligibility audit was performed without generating a v4 risk score.
The locked pretreatment-only rule retained 63 patients but only 15 deaths, so
PRINCE failed its minimum-event requirement and cannot serve as the primary
fresh validation. During the ensuing source audit, the official relocated ICGC
Release 28 open bucket was found to contain PACA-CA donor survival metadata and
sequencing-based expression. PACA-CA has not contributed to any v1-v4 model
selection, coefficient estimation, or reliability analysis and is distinct
from the PACA-AU development domain.

Before downloading or scoring the complete PACA-CA expression cohort, its
primary-tumour eligibility, untreated-specimen rule, deterministic one-profile-
per-donor choice, minimum sample/event counts, frozen-gene coverage, and
identifier-overlap audit are locked in
`V4_FRESH_CANDIDATES_AMENDMENT_PACA_CA.lock`. No PACA-CA association result may
change the frozen v4 bundle or select another algorithm. GSE77435 remains the
small orthogonal extreme-survivor test, and every ineligible or unfavorable
result remains reportable.

The subsequent source audit found that the official GSE77435 record explicitly
identifies its pancreatic tumour expression profiles as included in the earlier
GSE62452 submission. Because GSE62452 is a v4 development domain, GSE77435 is
permanently ineligible as an independent outcome validation regardless of its
different GEO accessions. `V4_GSE77435_SOURCE_OVERLAP_AMENDMENT.lock` permits
only outcome-free cross-platform reproducibility after an exact patient map is
established. Its survival labels cannot contribute to a fresh-validation claim.

### PACA-CA one-time analysis lock

PACA-CA passed the pre-score eligibility gate with 168 independent donors, 136
deaths, no detected development overlap, 98.9% or greater frozen-universe
coverage, and complete selected-gene and anchor coverage. Before loading the
v4 bundle on this cohort, the continuous-risk primary analysis, robust and
adjusted Cox models, fixed clinical cross-fitting, predecessor comparisons,
subgroups, 2,000 patient bootstraps, and Green/Amber/Red interpretation were
hashed in `V4_PACA_CA_ONE_TIME_ANALYSIS_PLAN.lock`. The separate computational
top-tier readiness rubric additionally requires improvement over the frozen
v2 predecessor and incremental value beyond a cross-fitted clinical model.
PACA-CA becomes permanently unavailable for tuning as soon as the locked script
is executed, irrespective of its result.
