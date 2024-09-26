from pathlib import Path
import json

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


work_dirs = Path('work_dirs')
for snap in work_dirs.rglob('**/epoch_300.pth'):
    for fn in sorted(snap.parent.rglob('**/*.json'), reverse=True):
        if fn.name == fn.parent.name + '.json':
            break
    else:
        continue
    stats = json.load(fn.open())
    if "test/locsim/AP" not in stats:
        continue
    print(fn)

    parts = snap.parent.name.split('_')
    name = 'yolox-' + parts[1]
    res = int(parts[3])

    row = [name, f'${res} \times {res}$', gflops[name, res], ]
    str(fn)