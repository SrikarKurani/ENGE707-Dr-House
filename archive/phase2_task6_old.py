import json, time, warnings
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from skeLCS import eLCS

warnings.filterwarnings('ignore')

# Task 6: model comparison (patched version).
# Changes from the earlier script:
#  1. Task 3 preprocessing is fitted on TRAINING rows only (no leakage from the test rows).
#  2. The eLCS settings for the improved system are chosen by 3-fold CV on the training set,
#     not by looking at test-set scores.
#  3. Holm correction added to the McNemar tests; extra raw-data LCS run with leakage columns removed.
TARGET, SEED = 'hospdead', 42
RAW_PATH = 'data/raw/support2.csv'
LEAKAGE_RAW = ['death', 'd.time', 'slos', 'surv2m', 'surv6m', 'prg2m', 'prg6m', 'sfdm2', 'hday']
CAT_RAW = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca', 'dnr']
ELCS_GRID = [dict(learning_iterations=5000, N=1000),     # library default N
             dict(learning_iterations=10000, N=1000),
             dict(learning_iterations=10000, N=2000),
             dict(learning_iterations=20000, N=2000)]


def log(*a):
    print(*a, flush=True)


class Preprocessor:
    """Task 3 cleaning as fit/transform: medians, flags and dummy levels come from the fitting rows only."""
    ZERO_INVALID = ['meanbp', 'hrt', 'resp']
    NEG_INVALID = ['totmcst', 'dnrday']
    NOMINAL = ['dzgroup', 'dzclass', 'race', 'ca', 'dnr']
    INCOME_MAP = {l: i for i, l in enumerate(['under $11k', '$11-$25k', '$25-$50k', '>$50k'])}

    def _base(self, df):
        d = df.drop(columns=[c for c in LEAKAGE_RAW + [TARGET] if c in df.columns]).copy()
        d = d.rename(columns={'num.co': 'num_co'})
        for c in self.ZERO_INVALID:
            d.loc[d[c] == 0, c] = np.nan
        for c in self.NEG_INVALID:
            d.loc[d[c] < 0, c] = np.nan
        return d

    def fit(self, df):
        d = self._base(df)
        self.num_cols = d.select_dtypes(include='number').columns.tolist()
        self.medians = d[self.num_cols].median()
        self.flag_cols = [c for c in self.num_cols if d[c].isnull().mean() > 0.40]
        self.income_median = d['income'].map(self.INCOME_MAP).median()
        self.levels = {c: sorted(d[c].fillna('Missing').unique())[1:] for c in self.NOMINAL}  # drop_first
        self.columns = self.transform(df).columns.tolist()
        return self

    def transform(self, df):
        d = self._base(df)
        out = pd.DataFrame(index=d.index)
        for c in self.num_cols:
            if c in self.flag_cols:
                out[f'{c}_was_missing'] = d[c].isnull().astype(int)
            out[c] = d[c].fillna(self.medians[c])
        out['sex'] = d['sex'].map({'male': 0, 'female': 1})
        out['income_was_missing'] = d['income'].isnull().astype(int)
        out['income'] = d['income'].map(self.INCOME_MAP).fillna(self.income_median).astype(int)
        for c in self.NOMINAL:
            v = d[c].fillna('Missing')
            for lvl in self.levels[c]:
                out[f'{c}_{lvl}'] = (v == lvl).astype(int)
        assert not out.isnull().any().any(), 'NaNs left after preprocessing'
        return out


raw = pd.read_csv(RAW_PATH)
y = raw[TARGET].values
tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=SEED, stratify=y)
raw_tr, raw_te = raw.iloc[tr].reset_index(drop=True), raw.iloc[te].reset_index(drop=True)
y_tr, y_te = y[tr], y[te]
log('raw', raw.shape, '| train', len(tr), '| test', len(te), '| positive rate', round(y.mean(), 4))


def label_encode(df, cols):
    df = df.copy()
    for c in cols:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))   # label->integer map only, no statistics learned
    return df


# raw / minimally processed inputs (Task 2 style; NaNs left for eLCS)
X_raw_leak = label_encode(raw, CAT_RAW + ['sfdm2']).drop(columns=[TARGET]).values
X_raw_noleak = label_encode(raw.drop(columns=LEAKAGE_RAW), CAT_RAW).drop(columns=[TARGET]).values

# cleaned inputs: preprocessing fitted on the training rows only
pre = Preprocessor().fit(raw_tr)
feature_names = pre.columns
Xc_tr = pre.transform(raw_tr).values.astype(float)
Xc_te = pre.transform(raw_te).values.astype(float)
log('cleaned feature count:', len(feature_names))

# ---- eLCS configuration search on the TRAINING set only ----
skf = StratifiedKFold(3, shuffle=True, random_state=SEED)
log('\n== eLCS config search (3-fold CV on training set, balanced accuracy) ==')
rows = []
for cfg in ELCS_GRID:
    sc, t0 = [], time.time()
    for a, b in skf.split(raw_tr, y_tr):
        p = Preprocessor().fit(raw_tr.iloc[a])          # preprocessing re-fitted inside every fold
        m = eLCS(random_state=SEED, **cfg).fit(p.transform(raw_tr.iloc[a]).values.astype(float), y_tr[a])
        sc.append(balanced_accuracy_score(y_tr[b], m.predict(p.transform(raw_tr.iloc[b]).values.astype(float))))
    rows.append({**cfg, 'cv_balacc_mean': np.mean(sc), 'cv_balacc_std': np.std(sc), 'secs': round(time.time() - t0, 1)})
    log(rows[-1])
cv_tab = pd.DataFrame(rows)
cv_tab.round(4).to_csv('task6_elcs_tuning_cv.csv', index=False)
best = {k: int(cv_tab.loc[cv_tab.cv_balacc_mean.idxmax(), k]) for k in ['learning_iterations', 'N']}
log('selected eLCS config:', best)

# ---- models ----
experiments = {
    'eLCS original (raw data, leakage kept)': (eLCS(learning_iterations=5000, random_state=SEED), X_raw_leak[tr], X_raw_leak[te]),
    'eLCS original (raw data, leakage removed)': (eLCS(learning_iterations=5000, random_state=SEED), X_raw_noleak[tr], X_raw_noleak[te]),
    'eLCS original (cleaned data)': (eLCS(learning_iterations=5000, random_state=SEED), Xc_tr, Xc_te),
    'eLCS improved (cleaned data)': (eLCS(random_state=SEED, **best), Xc_tr, Xc_te),
    'Random Forest': (RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1), Xc_tr, Xc_te),
    'Logistic Regression': (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), Xc_tr, Xc_te),
    'SVM (RBF)': (make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEED)), Xc_tr, Xc_te),
    'Decision Tree': (DecisionTreeClassifier(random_state=SEED), Xc_tr, Xc_te),
}
rows, preds, fitted = [], {}, {}
for name, (model, Xa, Xb) in experiments.items():
    t0 = time.time(); model.fit(Xa, y_tr); secs = time.time() - t0
    pred, proba = model.predict(Xb), model.predict_proba(Xb)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    rows.append({'Model': name, 'Acc': accuracy_score(y_te, pred), 'BalAcc': balanced_accuracy_score(y_te, pred),
                 'Prec': precision_score(y_te, pred, zero_division=0), 'Recall': recall_score(y_te, pred),
                 'F1': f1_score(y_te, pred), 'ROC-AUC': roc_auc_score(y_te, proba),
                 'PR-AUC': average_precision_score(y_te, proba), 'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
                 'Train s': round(secs, 1)})
    preds[name], fitted[name] = pred, model
    log(f'done: {name} ({secs:.1f}s)')
results = pd.DataFrame(rows).set_index('Model')
results.round(4).to_csv('task6_results.csv')
log('\n' + results.round(4).to_string())

# ---- exact McNemar vs improved LCS, Holm-corrected ----
ref = 'eLCS improved (cleaned data)'
mc = []
for name in preds:
    if name == ref:
        continue
    a_ok, b_ok = preds[ref] == y_te, preds[name] == y_te
    n01, n10 = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    p = 1.0 if n01 + n10 == 0 else binomtest(min(n01, n10), n01 + n10, 0.5).pvalue
    mc.append({'Compared with': name, 'improved_only_correct': n01, 'other_only_correct': n10, 'p_raw': p})
mc = pd.DataFrame(mc)
order, adj, run = np.argsort(mc.p_raw.values), np.empty(len(mc)), 0.0
for rank, i in enumerate(order):
    run = max(run, (len(mc) - rank) * mc.p_raw.values[i]); adj[i] = min(1.0, run)
mc['p_holm'] = adj
mc.to_csv('task6_mcnemar.csv', index=False)
log(f'\nMcNemar (exact) vs {ref}, Holm-corrected\n' + mc.to_string(index=False))

# ---- Task 7: rules from the improved eLCS (only conditions that restrict the observed range are shown) ----
pop = fitted[ref].population.popSet
lo, hi = Xc_tr.min(axis=0), Xc_tr.max(axis=0)
rule_rows = []
for cl in pop:
    parts = []
    for att, c in zip(cl.specifiedAttList, cl.condition):
        n = feature_names[att]
        if isinstance(c, (list, tuple, np.ndarray)):
            l, h = float(c[0]), float(c[1]); hl, hh = l > lo[att], h < hi[att]
            if hl and hh: parts.append(f'{l:.3g} <= {n} <= {h:.3g}')
            elif hl: parts.append(f'{n} >= {l:.3g}')
            elif hh: parts.append(f'{n} <= {h:.3g}')
        else:
            parts.append(f'{n} == {float(c):g}')
    rule_rows.append({'class': int(cl.phenotype), 'numerosity': cl.numerosity, 'fitness': cl.fitness,
                      'accuracy': cl.accuracy, 'match_count': cl.matchCount, 'correct_count': cl.correctCount,
                      'n_conditions_total': len(cl.specifiedAttList), 'n_conditions_effective': len(parts),
                      'rule': ' AND '.join(parts)})
rules = pd.DataFrame(rule_rows)
rules['support_score'] = rules.numerosity * rules.fitness
rules.sort_values('support_score', ascending=False).to_csv('task7_all_rules.csv', index=False)
log(f'\nPopulation: {len(pop)} macro / {int(rules.numerosity.sum())} micro | mean effective conditions '
    f'{rules.n_conditions_effective.mean():.1f} | class1 {int((rules["class"]==1).sum())} class0 {int((rules["class"]==0).sum())}')
with open('task6_run_info.json', 'w') as f:
    json.dump({'selected_elcs_config': best, 'split': 'stratified 80/20, random_state=42',
               'n_train': int(len(tr)), 'n_test': int(len(te)),
               'preprocessing': 'Task 3 steps fitted on training rows only'}, f, indent=2)
log('ALL DONE')
