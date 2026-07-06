"""
Run PRIDICT (schwank_rnnattn v3, KLDloss 3-outcome model) on our 13,844-pegRNA
target-disjoint test split. Predicts averageedited / averageunedited / averageindel
per pegRNA and reports Spearman and Pearson correlations against ground truth.

Notes
-----
* Test pegRNAs are joined to the PRIDICT supplementary Excel (Library_1, 92,423 rows)
  on (initial_sequence == wide_initial_target) AND (mutated_sequence == wide_mutated_target).
  This supplies all engineered features (MFE, melting temps, GC, lengths, locations)
  that the model needs.
* Predictions are averaged across all 5 PRIDICT runs (run_0..run_4) for the v3 split.
* Leakage caveat: PRIDICT was trained on Library_1 with its own random split, so
  some of these test pegRNAs were almost certainly in PRIDICT's training set.
  We do NOT filter for that here; we only report correlations.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, '/tmp/pridict_check')

# pandas >= 1.5 makes pd.get_dummies return bool by default, which then
# pollutes the seqlevel feature matrix to object dtype when concatenated
# with numeric float columns. Force the legacy uint8 behavior so the dataset
# code (which calls torch.from_numpy on the slice) keeps working.
import pandas as _pd
_orig_get_dummies = _pd.get_dummies
def _get_dummies_int(*args, **kwargs):
    kwargs.setdefault('dtype', 'uint8')
    return _orig_get_dummies(*args, **kwargs)
_pd.get_dummies = _get_dummies_int

from prieml.predict_outcomedistrib import PRIEML_Model
from prieml.utilities import get_device, compute_pearson_corr, compute_spearman_corr
from scipy import stats as scipy_stats


TEST_CSV = '/Users/furkanozden/Desktop/crispAIPE/pe-uncert/data/pridict_data_revs/pridict-test_revs.csv'
SUPP_XLSX = '/Users/furkanozden/Desktop/crispAIPE/pe-uncert/data/pridict_data/41587_2022_1613_MOESM5_ESM.xlsx'
MODEL_DIR = '/tmp/pridict_check/trained_models/schwank_rnnattn/v3/train_val'
OUT_DIR = '/Users/furkanozden/Desktop/crispAIPE/pe-uncert/test/figures_revs/performance_comparison'

Y_REF = ['averageedited', 'averageunedited', 'averageindel']


def build_input_df():
    print('--- loading test set ---')
    test = pd.read_csv(TEST_CSV)
    print('  test shape:', test.shape)

    print('--- loading supplementary Library_1 ---')
    supp = pd.read_excel(SUPP_XLSX, sheet_name='Library_1')
    print('  supp shape:', supp.shape)

    # Normalize case for the join keys
    test['_init'] = test['initial_sequence'].str.upper()
    test['_mut']  = test['mutated_sequence'].str.upper()
    supp['_init'] = supp['wide_initial_target'].str.upper()
    supp['_mut']  = supp['wide_mutated_target'].str.upper()

    # Sanity: supp must be unique on (init, mut)
    n_unique = supp.drop_duplicates(subset=['_init', '_mut']).shape[0]
    assert n_unique == supp.shape[0], (
        f'supp not unique on (init,mut): {n_unique} vs {supp.shape[0]}'
    )

    # Columns we need from supp for PRIDICT inference + ground truth check
    cols_from_supp = [
        '_init', '_mut',
        'wide_initial_target', 'wide_mutated_target',
        'protospacerlocation_only_initial', 'PBSlocation',
        'RT_initial_location', 'RT_mutated_location',
        'Correction_Type',
        # length features
        'Correction_Length', 'RToverhangmatches', 'RToverhanglength', 'RTlength', 'PBSlength',
        # MFE features
        'MFE_protospacer', 'MFE_protospacer_scaffold', 'MFE_extension',
        'MFE_extension_scaffold', 'MFE_protospacer_extension_scaffold',
        'MFE_rt', 'MFE_pbs',
        # melting temps
        'RTmt', 'RToverhangmt', 'PBSmt', 'protospacermt', 'extensionmt',
        'original_base_mt', 'edited_base_mt',
        # supp ground truth (for sanity check against our test set)
        'averageedited', 'averageunedited', 'averageindel',
    ]
    supp_sub = supp[cols_from_supp].copy()

    merged = test.merge(
        supp_sub,
        left_on=['_init', '_mut'],
        right_on=['_init', '_mut'],
        how='left',
        indicator=True,
    )
    vc = merged['_merge'].value_counts().to_dict()
    print('  merge counts:', vc)
    assert vc.get('left_only', 0) == 0, 'some test pegRNAs failed to join to supp'
    assert merged.shape[0] == test.shape[0], 'merge changed row count'

    # melting-temp nan indicators (notebook sets them to 0)
    merged['original_base_mt_nan'] = 0.0
    merged['edited_base_mt_nan']   = 0.0

    # Sanity check: supp ground truth should be in [0,1] and our test
    # ground truth (edited_percentage etc.) is in [0,100].
    print('  supp averageedited range:',
          merged['averageedited'].min(), merged['averageedited'].max())
    print('  test edited_percentage range:',
          merged['edited_percentage'].min(), merged['edited_percentage'].max())

    # Stable seq_id keyed by row order so we can re-align predictions
    merged.reset_index(drop=True, inplace=True)
    merged['seq_id'] = [f'seq_{i}' for i in range(merged.shape[0])]
    return merged


def main():
    df = build_input_df()
    n = df.shape[0]
    print(f'--- N = {n} pegRNAs ---')

    device = get_device(False, 0)
    print('device:', device)

    prieml_model = PRIEML_Model(device, wsize=20, normalize='max', fdtype=torch.float32)

    # build dataloader once; reuse across 5 runs
    print('--- preparing dataloader ---')
    dloader = prieml_model.prepare_data(df, y_ref=Y_REF, batch_size=512)

    per_run_preds = []
    for run in range(5):
        mdir = os.path.join(MODEL_DIR, f'run_{run}')
        print(f'--- predicting run_{run} ---')
        pred_df = prieml_model.predict_from_dloader(dloader, mdir, y_ref=Y_REF)
        per_run_preds.append(pred_df)

    # Concatenate then average across runs by seq_id
    all_pred = pd.concat(per_run_preds, axis=0, ignore_index=True)
    avg_pred = prieml_model.compute_avg_predictions(all_pred)

    # Re-align with df order via seq_id
    df_sorted = df[['seq_id']].copy()
    aligned = df_sorted.merge(avg_pred, on='seq_id', how='left')
    assert aligned['pred_averageedited'].notna().all(), 'missing predictions after alignment'

    pred = aligned[[f'pred_{t}' for t in Y_REF]].to_numpy(dtype=np.float64)
    # Convert test ground truth from 0-100 to 0-1 so it lives on the same scale
    # as PRIDICT predictions. (Correlations don't care but easier to inspect.)
    true = df[['edited_percentage', 'unedited_percentage', 'indel_percentage']].to_numpy(dtype=np.float64) / 100.0
    seq_ids = df['seq_id'].to_numpy().astype('U16')

    print('--- predictions shape:', pred.shape, ' true shape:', true.shape)
    print('--- prediction sums (should be ~1 since KLD over 3 outcomes):',
          pred.sum(axis=1)[:5])
    print('--- true sums (~1 expected if read counts conserved):',
          true.sum(axis=1)[:5])

    # Correlations per outcome
    out = {'n_samples': int(n)}
    labels = ['edited', 'unedited', 'indel']
    for i, lab in enumerate(labels):
        sp, _ = scipy_stats.spearmanr(pred[:, i], true[:, i])
        pe, _ = scipy_stats.pearsonr(pred[:, i], true[:, i])
        out[f'{lab}_spearman'] = float(sp)
        out[f'{lab}_pearson']  = float(pe)
        print(f'  {lab}: spearman={sp:.4f}  pearson={pe:.4f}')

    # Also report against supp ground truth (sanity: should be the same up to noise
    # since our test ground truth was derived from the same library reads)
    supp_true = df[['averageedited', 'averageunedited', 'averageindel']].to_numpy(dtype=np.float64)
    print('  supp-vs-test correlation (sanity):')
    for i, lab in enumerate(labels):
        sp, _ = scipy_stats.spearmanr(true[:, i], supp_true[:, i])
        print(f'    {lab}: spearman test_vs_supp={sp:.4f}')

    npz_path = os.path.join(OUT_DIR, 'pridict_predictions_targetdisjoint.npz')
    json_path = os.path.join(OUT_DIR, 'pridict_targetdisjoint_correlations.json')
    np.savez(npz_path, pred=pred, true=true, seq_ids=seq_ids)
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)
    print('saved:', npz_path)
    print('saved:', json_path)


if __name__ == '__main__':
    main()
