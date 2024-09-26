from pathlib import Path
import json
from matplotlib import pyplot as plt

gflops = {
        ('yolox-tiny', 640): 10.3,
        ('yolox-s', 640): 18.3,
        ('yolox-m', 640): 47.9,
        ('yolox-l', 640): 97.5,
        ('yolox-tiny', 960): 23.3,
        ('yolox-s', 960): 41.1,
        ('yolox-m', 960): 108.0,
        ('yolox-l', 960): 219.3,
        ('yolox-tiny', 1280): 41.5,
        ('yolox-s', 1280): 73.2,
        ('yolox-m', 1280): 192.0,
        ('yolox-l', 1280): 390.0,
}


class Resolution(int):
    pass

work_dirs = Path('work_dirs')
rows = []
for snap in work_dirs.rglob('**/epoch_300.pth'):
    for fn in sorted(snap.parent.rglob('**/*.json'), reverse=True):
        if fn.name == fn.parent.name + '.json':
            break
    else:
        continue
    stats = json.load(fn.open())
    if "test/locsim/AP" not in stats:
        continue

    parts = snap.parent.name.split('_')
    name = 'yolox-' + parts[1]
    res = int(parts[3])

    stats.keys()

    common = [Resolution(res), gflops[name, res], stats['test/bbox/AP']*100]
    rows.append([name + ' bbox'] + common + [stats['test/locsim_bbox/AP']*100, stats['test/locsim_bbox/precision']*100, stats['test/locsim_bbox/recall']*100, stats['test/locsim_bbox/f1']*100])
    rows.append([name + ' pose'] + common + [stats['test/locsim/AP']*100, stats['test/locsim/precision']*100, stats['test/locsim/recall']*100, stats['test/locsim/f1']*100])

def fmt(val):
    if isinstance(val, str):
        return val
    elif isinstance(val, Resolution):
        return f'${val} \\times {val}$'

    return f'{val:.1f}'

def skey(r):
    return (r[0].split(' ')[1], r[1], r[2])

rows.sort(key=skey)
prev_res = None
for r in rows:
    if prev_res is not None and prev_res != r[1]:
        print('        \\vspace{-3mm} \\\\\n')
    print('        ' + ' & '.join(fmt(c) for c in r) + '\\\\')
    prev_res = r[1]

def plot_set(name, res, style, color):
    rr = [r for r in rows if name in r[0] and r[1]==res]
    plt.plot([r[2] for r in rr], [r[4] for r in rr], style, label=f'{name} {res}x{res}', color=color)

colors = ['#d7191c','#fdae61','#2c7bb6','#abd9e9']

plt.clf()
for i, res in enumerate(sorted(set([r[1] for r in rows]), reverse=True)):
    plot_set('pose', res, '-o', colors[i])
for i, res in enumerate(sorted(set([r[1] for r in rows]), reverse=True)):
    plot_set('bbox', res, '-o', colors[i + 2])
plt.legend()
plt.xlabel('GFLOPS')
plt.ylabel('mAP-LocSim')
plt.savefig('locsim.pdf')
