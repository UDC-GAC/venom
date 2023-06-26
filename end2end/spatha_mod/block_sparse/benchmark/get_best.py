import pandas as pd
from math import log10, floor


def round_to_1(x):
    return round(x, -int(floor(log10(abs(x)))))

df = pd.read_csv('/tmp/transformer.csv', names=['arch', 'm', 'k', 'vrow', 'vcol', 'density', 'seed', 'order', 'n', 'bm', 'bn', 'bk', 'wm', 'wn', 'wk', 'mm', 'mn', 'mk', 'nstage', 'time', 'tflops'])
df['density'] = df['density'].apply(round_to_1)
df = df.groupby(['arch','m','k','vrow','vcol','density','order','n','bm','bn','bk','wm','wk','wn','mm','mn','mk','nstage']).mean().reset_index()

cfg = df.sort_values(by='time')
cfg = cfg.groupby(['arch','m','k','vrow','vcol','density','n']).apply(pd.DataFrame.head, n=1)
#cfg = cfg.groupby(['arch','m','k','density','n']).apply(pd.DataFrame.head, n=1)
cfg = cfg.drop(['order', 'seed'], axis=1)

pd.set_option('display.max_rows', 50000)
print(cfg[cfg.vrow==16])