import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', nargs='+', required=True)
    p.add_argument('--metrics', nargs='+', required=True)
    p.add_argument('--unit', nargs='+', default=['seed'])
    p.add_argument('--reference', required=True)
    p.add_argument('--bootstrap', type=int, default=10000)
    p.add_argument('--seed', type=int, default=13)
    p.add_argument('--output', default='runs/comparison.csv')
    args = p.parse_args()
    frames = []
    for pattern in args.inputs:
        for file in sorted(glob.glob(pattern)):
            frame = pd.read_csv(file)
            settings = Path(file).parent / 'settings.json'
            if 'seed' not in frame and settings.exists():
                frame['seed'] = json.loads(settings.read_text(encoding='utf-8'))['seed']
            required = {'method', *args.unit, *args.metrics}
            if not required <= set(frame):
                raise ValueError(str(file) + ' missing columns ' + str(required - set(frame)))
            frames.append(frame)
    if not frames:
        raise ValueError('No input metric tables matched')
    data = pd.concat(frames, ignore_index=True)
    data = data.groupby(['method', *args.unit], dropna=False)[args.metrics].mean().reset_index()
    data['unit_id'] = data[args.unit].astype(str).agg('|'.join, axis=1)
    rng = np.random.default_rng(args.seed)
    rows = []
    for metric in args.metrics:
        wide = data.pivot(index='unit_id', columns='method', values=metric)
        if args.reference not in wide:
            raise ValueError('Reference method absent: ' + args.reference)
        family = []
        for method in wide:
            values = wide[method].dropna().to_numpy()
            if len(values) < 2:
                low, high, sd = np.nan, np.nan, np.nan
            else:
                resampled = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(args.bootstrap)])
                low, high = np.quantile(resampled, [.025, .975])
                sd = values.std(ddof=1)
            row = dict(metric=metric, method=method, units=len(values), mean=values.mean(), sd=sd, ci_low=low, ci_high=high)
            if method != args.reference:
                paired = wide[[args.reference, method]].dropna()
                difference = (paired[args.reference] - paired[method]).to_numpy()
                row['paired_units'] = len(difference)
                if len(difference) >= 2:
                    boot = np.array([rng.choice(difference, len(difference), replace=True).mean() for _ in range(args.bootstrap)])
                    row.update(reference_minus_method=difference.mean(), difference_ci_low=np.quantile(boot, .025), difference_ci_high=np.quantile(boot, .975))
                    nonzero = difference[difference != 0]
                    p_value = float(wilcoxon(difference).pvalue) if len(nonzero) else 1.
                    ranks = rankdata(abs(nonzero))
                    row['rank_biserial'] = float(np.sum(ranks * np.sign(nonzero)) / ranks.sum()) if len(ranks) else 0.
                    walsh = (difference[:, None] + difference[None]) / 2
                    row['hodges_lehmann'] = float(np.median(walsh[np.triu_indices(len(difference))]))
                    row['p_raw'] = p_value
                    family.append(row)
            rows.append(row)
        ordered = sorted(family, key=lambda row: row['p_raw'])
        running = 0.
        for index, row in enumerate(ordered):
            running = max(running, min(1., (len(ordered) - index) * row['p_raw']))
            row['p_holm'] = running
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == '__main__':
    main()
