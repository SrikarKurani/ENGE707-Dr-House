# Task 7: Interpretation of Results and Explainable AI

## 7.1 Interpretation of the main findings

All eight systems were trained on the same stratified 80/20 split (random_state = 42; 7,284 training and 1,821 test records) and scored on the same test rows. The positive class is in-hospital death (`hospdead` = 1), which makes up 472 of the 1,821 test records (about 26%). The Task 3 preprocessing (medians, missing-value flags and category levels) was fitted on the training rows only and then applied to the test rows. The improved LCS is eLCS with 10,000 learning iterations and a rule population of N = 2000. These settings were chosen by 3-fold cross-validation on the training set (mean balanced accuracy 0.800, against 0.769 for the default 5,000 iterations with N = 1000), and the test set was not used for the choice.

**Table 7.1: Test-set performance**

| System | Acc. | Bal. acc. | Prec. | Recall | F1 | ROC-AUC | PR-AUC | Train time (s) |
|---|---|---|---|---|---|---|---|---|
| eLCS original (raw data, leakage columns kept) | 0.925 | 0.905 | 0.850 | 0.862 | 0.856 | 0.959 | 0.872 | 6.1 |
| eLCS original (raw data, leakage columns removed) | 0.878 | 0.792 | 0.881 | 0.612 | 0.723 | 0.890 | 0.821 | 5.9 |
| eLCS original (cleaned data) | 0.847 | 0.731 | 0.862 | 0.489 | 0.624 | 0.874 | 0.756 | 8.6 |
| eLCS improved (cleaned data) | 0.881 | 0.808 | 0.854 | 0.655 | 0.741 | 0.908 | 0.805 | 38.0 |
| Random Forest | 0.912 | 0.874 | 0.856 | 0.795 | 0.824 | 0.955 | 0.901 | 3.7 |
| Logistic Regression | 0.908 | 0.862 | 0.864 | 0.767 | 0.813 | 0.947 | 0.889 | 0.0 |
| SVM (RBF) | 0.903 | 0.856 | 0.854 | 0.756 | 0.802 | 0.948 | 0.885 | 7.4 |
| Decision Tree | 0.852 | 0.817 | 0.703 | 0.744 | 0.723 | 0.817 | 0.590 | 0.1 |

**Statistical comparison.** Each system was compared with the improved LCS using an exact McNemar test on the paired test-set predictions, with a Holm correction for the seven comparisons.

**Table 7.2: Exact McNemar tests against the improved LCS**

| Compared system | Improved LCS only correct | Other system only correct | p (raw) | p (Holm) |
|---|---|---|---|---|
| eLCS original (raw data, leakage kept) | 57 | 136 | 1.2 × 10⁻⁸ | 8.7 × 10⁻⁸ |
| eLCS original (raw data, leakage removed) | 48 | 42 | 0.598 | 0.598 |
| eLCS original (cleaned data) | 109 | 47 | 7.5 × 10⁻⁷ | 3.8 × 10⁻⁶ |
| Random Forest | 26 | 82 | 6.1 × 10⁻⁸ | 3.7 × 10⁻⁷ |
| Logistic Regression | 33 | 82 | 5.6 × 10⁻⁶ | 2.2 × 10⁻⁵ |
| SVM (RBF) | 31 | 71 | 9.3 × 10⁻⁵ | 2.8 × 10⁻⁴ |
| Decision Tree | 161 | 108 | 1.5 × 10⁻³ | 3.0 × 10⁻³ |

Every difference is significant at the 5% level after correction except the one against the original LCS on raw data with the leakage columns removed. The test compares only the 1,821 test predictions, so it reflects one data split and does not measure variation across different training sets.

**What the comparison shows.**

1. *The improved LCS is clearly better than the original LCS on the same cleaned data.* Balanced accuracy rose from 0.731 to 0.808, recall from 0.489 to 0.655, F1 from 0.624 to 0.741, ROC-AUC from 0.874 to 0.908 and PR-AUC from 0.756 to 0.805. The McNemar test agrees (109 test records only the improved system classified correctly, against 47; Holm p = 3.8 × 10⁻⁶). The cost is about four times the training time (38.0 s against 8.6 s).
2. *Part of this gain recovers performance lost in preprocessing.* The original LCS run on the raw data with the leakage columns removed, where missing values were left for eLCS to handle itself, reaches a balanced accuracy of 0.792. The improved LCS (0.808) is not significantly different from it (Holm p = 0.598). The original LCS gets worse (0.792 to 0.731) once missing values are filled with medians and categories are one-hot encoded. This suggests that the imputation and encoding choices of Task 3 do not suit the original LCS, and the improved settings (a larger rule population and more iterations) mostly compensate for that. This is a hypothesis; we did not test it separately.
3. *The improved LCS does not match the conventional models.* Its balanced accuracy is 0.066 below Random Forest, 0.055 below Logistic Regression and 0.048 below the SVM, and the McNemar tests show the conventional models classify significantly more test records correctly. Its precision (0.854) is similar to theirs (0.854 to 0.864), but its recall (0.655) is much lower than Random Forest's (0.795). It therefore misses about one in three patients who die (163 false negatives out of 472) while raising few false alarms (53). All conventional models were run with default settings and no class weighting, as the LCS was, so the gap is not due to imbalance handling that only they received.
4. *Against the Decision Tree the picture is mixed.* The improved LCS is better on accuracy (0.881 against 0.852), precision (0.854 against 0.703), ROC-AUC (0.908 against 0.817) and PR-AUC (0.805 against 0.590), and it classifies more test records correctly (161 against 108; Holm p = 0.003). The Decision Tree is slightly higher on balanced accuracy (0.817 against 0.808) because of its higher recall.
5. *The original LCS on raw data with the leakage columns kept scores highest on balanced accuracy and recall, but this is not a like-for-like result.* That run includes columns identified in Phase I as leakage (for example `d.time`, `slos`, `sfdm2` and the survival-estimate columns). Removing them lowers its balanced accuracy from 0.905 to 0.792, so the headline figure mainly measures how much information those columns carry about the outcome. On leakage-free data, Random Forest actually exceeds it on PR-AUC (0.901 against 0.872).

## 7.2 Important rules, feature conditions and classifier patterns

To study the decision-making of the improved LCS, the rule population of the model above was exported. It contains 1,966 distinct rules (2,000 including numerosity), of which 812 predict in-hospital death and 1,154 predict survival to discharge. Rules are long: they average 20 conditions (minimum 9, maximum 39). Because eLCS learns interval bounds that can extend beyond the observed data range, bounds that reach or pass the observed minimum or maximum are written below as one-sided conditions (for example `bun <= 91.8`).

**Most-used features.** Counting each condition once per rule and weighting by numerosity, the most common features, out of a total numerosity of 2,000, are `dnr_dnr before sadm` (1,103), `race_other` (1,095), `dzgroup_Cirrhosis` (1,086), `dnr_dnr after sadm` (1,079), `dnr_no dnr` (1,052), the treatment-intensity score `avtisst` (999), `dzgroup_COPD` (936), `dementia` (919), `race_asian` (905) and `dzgroup_Coma` (887). Frequent use does not mean importance. Many of the frequent conditions are of the form "indicator = 0" on a category that applies to few patients (such as a race group or a rare diagnosis), so the condition excludes very few patients and probably reflects over-general rules, not a meaningful pattern. The DNR indicators and `avtisst` are the exception, since they also appear in the sample rules below.

### Sample rules

Three rules were chosen to show different decision patterns. "Accuracy" is the rule's correct-prediction rate over the training instances it matched, and "matches" is the number of those instances.

**Rule A: a rule predicting in-hospital death (12 conditions; accuracy 0.916, 175 correct of 191 matches).**

> IF `avtisst` >= 23.5 AND `dnr_dnr after sadm` == 1 AND 30.5 <= `aps` <= 101 AND `dementia` == 0 AND `adlp` == 0 AND `glucose_was_missing` == 1 AND `bun` <= 91.8 AND `totcst` <= 1.58 × 10⁵ AND `charges` <= 4.58 × 10⁵ AND `dzgroup_COPD` == 0 AND `dzgroup_Cirrhosis` == 0 AND `dzgroup_Colon Cancer` == 0 THEN `hospdead` = 1

The central conditions are a DNR order written after admission, a high treatment-intensity score (`avtisst` of at least 23.5) and an acute physiology score (`aps`) of at least 30.5. The remaining conditions exclude dementia, three diagnosis groups, and extremes of urea and cost, and include flags that glucose was not measured (`glucose_was_missing`) and that the activities-of-daily-living score `adlp` equals 0. The pattern describes a patient who is receiving intensive treatment and has a moderate-to-high physiological severity, and for whom a DNR decision was made during the stay. A care team making that decision is signalling a poor prognosis, so the predicted death is plausible clinically. The rule is accurate but narrow (191 matches).

**Rule B: a confident rule predicting survival (11 conditions; accuracy 0.970, 224 correct of 231 matches).**

> IF `dnr_no dnr` == 1 AND `dnr_dnr after sadm` == 0 AND `avtisst` <= 30.5 AND `crea` <= 7.72 AND `sex` == 0 (male) AND `adlp_was_missing` == 1 AND `dzgroup_Coma` == 0 AND `dzgroup_Cirrhosis` == 0 AND `dzgroup_MOSF w/Malig` == 0 AND `dzclass_Cancer` == 0 AND `race_hispanic` == 0 THEN `hospdead` = 0

This rule describes patients with no DNR order at any point, only moderate treatment intensity (`avtisst` at most 30.5), no extreme kidney failure, and a diagnosis outside the coma, cirrhosis, cancer and multi-organ-failure-with-malignancy groups. It is the mirror image of Rule A: the absence of a DNR order and moderate care intensity suggest the care team did not consider the patient close to death. The conditions on sex and race are unlikely to be clinically meaningful and probably narrow the rule by chance, which is one reason why the rule is accurate but matches few patients.

**Rule C: a broad rule predicting survival (12 conditions; accuracy 0.963, 678 correct of 704 matches).**

> IF `dnr_dnr after sadm` == 0 AND `avtisst` <= 42.3 AND 7.09 <= `ph` <= 7.73 AND 34.9 <= `temp` <= 38.2 AND `resp` <= 47.4 AND `scoma` <= 34.5 AND `alb` <= 9.01 AND `charges` <= 3.48 × 10⁵ AND `dzgroup_Cirrhosis` == 0 AND `dzgroup_Colon Cancer` == 0 AND `race_asian` == 0 AND `race_other` == 0 THEN `hospdead` = 0

This is the most general of the three rules. It matches 704 training instances, more than three times as many as Rule A or B, and still has high accuracy. Its clinical conditions require no DNR order written after admission, moderate treatment intensity, body temperature within a normal-to-feverish range, and pH, respiratory rate and coma score without extreme values. It acts as a general "no acute deterioration" pattern. The two diagnosis exclusions and the race conditions probably restrict few patients.

Taken together, the three rules rely on the same small set of variables: DNR status and timing, treatment intensity (`avtisst`), acute physiology and diagnosis group. This agrees with the feature-usage counts above and shows that the model's decisions are driven by a few strongly outcome-related variables, with many remaining conditions acting as loose filters.

## 7.3 How LCS rule-based output supports interpretability

Each classifier is an explicit IF–THEN statement with attached statistics (accuracy, fitness, numerosity, match and correct counts). This supports interpretability in four ways:

- **Traceable predictions.** For any patient, the rules that match and their votes can be listed, so a prediction can be traced to specific conditions.
- **Local explanations.** Each rule describes one subgroup of patients, so different parts of the data are explained by different rules, in contrast to a single set of global weights.
- **Built-in reliability information.** Accuracy and match counts show which rules are well supported (Rule C) and which are narrower (Rules A and B).
- **Visible handling of missing data.** Missing-value flags such as `adlp_was_missing` and `glucose_was_missing` can appear in conditions, which exposes how missingness influences the model.

Interpretability is limited, however, by rule length and population size. With about 20 conditions per rule and about 2,000 rules voting together, a clinician can read any one rule but cannot grasp the model as a whole.

## 7.4 Are the explanations trustworthy and practically meaningful?

We regard the explanations as only partly trustworthy, for the following reasons.

1. **Over-general conditions.** Many conditions hold for almost every patient (for example wide intervals or "indicator = 0" for a rare category), so they inflate rule length without adding meaning.
2. **Unclipped intervals.** eLCS learns interval bounds that can fall outside possible values (for example negative bounds on a variable that cannot be negative). We removed these by showing only the bounds that restrict the observed range, but the original rules look more specific than they are.
3. **Possible proxy variables.** DNR status and timing, treatment intensity (`avtisst`), and the cost variables `totcst`, `totmcst` and `charges` are determined or accumulated during the hospital stay, and several physiology scores (`aps`, `sps`, `scoma`) are measured around day 3 of the stay. Rules based on them, particularly Rule A, may therefore reflect the care team's own judgement of the patient's prognosis and would not be available at admission. This should be checked against the data dictionary.
4. **Race and sex variables in the rules.** Race indicators are among the most frequently used conditions, and two of the sample rules contain race or sex conditions. Even if these are mostly uninformative, using such variables as inputs to a clinical prediction model raises fairness concerns and would need review before any practical use.
5. **Optimistic rule statistics.** Rule accuracies and match counts are accumulated during training, so they are not unbiased estimates of how reliable a rule will be on new patients.
6. **Association, not causation.** A rule such as "DNR after admission and high treatment intensity predicts death" describes a pattern in this dataset. It does not show that either factor causes the outcome and should not guide treatment decisions.
7. **A single model on a single dataset.** Different random seeds would give different rule populations, and the rules have not been validated externally or reviewed by a clinician.

Practically, the rules are useful for exploring what the model has learned and for identifying variables that domain experts should examine, but they are not reliable enough to be presented as clinical explanations. Rule compaction, restriction to variables known at prediction time, and removal of weak or sensitive variables would make the explanations shorter and more meaningful.

