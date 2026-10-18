"""Small deterministic KITTI sample, downloaded by HTTP ranges rather than a 12 GB ZIP."""
import hashlib
import io
import json
import random
import re
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

IMAGES_URL = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip'
LABELS_URL = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip'
SEED = 20261018

def sha256(data):
    return hashlib.sha256(data).hexdigest()

def download(url, maximum=50_000_000):
    with urllib.request.urlopen(url, timeout=90) as response:
        data = response.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError('Download exceeds size limit')
    return data

class RangeFile(io.RawIOBase):
    """Seekable, bounded remote ZIP reader. Refuses a server that ignores Range."""
    def __init__(self, url):
        self.url, self.position = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, headers={'Range':'bytes=-1'}), timeout=60) as r:
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', r.headers.get('Content-Range', ''))
            if r.status != 206 or not match:
                raise ValueError('Server does not support byte ranges')
            self.size = int(match[3])
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if self.position < 0: raise ValueError('Negative seek')
        return self.position
    def read(self, size=-1):
        size = min(self.size-self.position, size if size >= 0 else self.size-self.position)
        if size <= 0: return b''
        if size > 8_000_000: raise ValueError('Remote ZIP read exceeds size limit')
        start, end = self.position, self.position+size-1
        request = urllib.request.Request(self.url, headers={'Range':f'bytes={start}-{end}'})
        with urllib.request.urlopen(request, timeout=90) as r:
            if r.status != 206 or r.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}':
                raise ValueError('Unexpected byte-range response')
            result = r.read(size+1)
        if len(result) != size: raise ValueError('Truncated remote ZIP read')
        self.position += size
        return result

def labels(text):
    result=[]
    for line in text.splitlines():
        fields=line.split()
        if len(fields)!=15: raise ValueError('Expected KITTI 15-field labels')
        box=list(map(float,fields[4:8]))
        if not all(__import__('math').isfinite(v) for v in box) or box[2]<=box[0] or box[3]<=box[1]:
            raise ValueError('Invalid annotation box')
        result.append({'kind':fields[0], 'truncation':float(fields[1]), 'occlusion':int(fields[2]), 'box':box})
    return result

def prepare(root, per_group=32):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists(): raise ValueError('Sample already prepared; use another directory')
    if not isinstance(per_group,int) or not 1 <= per_group <= 256: raise ValueError('Invalid sample size')
    raw=download(LABELS_URL,10_000_000)
    all_labels={}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for name in sorted(z.namelist()):
            if re.fullmatch(r'training/label_2/\d{6}\.txt',name):
                all_labels[Path(name).stem]=labels(z.read(name).decode('utf-8'))
    rng=random.Random(SEED)
    positive=[k for k,v in all_labels.items() if any(a['kind']=='Pedestrian' for a in v)]
    negative=[k for k in all_labels if k not in set(positive)]
    selected=sorted(rng.sample(positive,per_group)+rng.sample(negative,per_group))
    # Freeze IDs and labels before any image inference, including all negative scenes.
    pending={'seed':SEED,'selection':'Equal positive/negative image strata; random sampling within each stratum',
             'ids':selected,'labels_archive_sha256':sha256(raw),'images_url':IMAGES_URL,'labels_url':LABELS_URL,
             'annotations':{k:all_labels[k] for k in selected},'images':{}}
    (root/'selection.json').write_text(json.dumps(pending,indent=2),encoding='utf-8')
    (root/'images').mkdir(exist_ok=True)
    def worker(ids):
        output={}
        with RangeFile(IMAGES_URL) as remote,zipfile.ZipFile(remote) as z:
            for key in ids:
                content=z.read(f'training/image_2/{key}.png')
                (root/'images'/f'{key}.png').write_bytes(content)
                output[key]=sha256(content)
                print(f'Downloaded KITTI {key}',flush=True)
        return output
    with ThreadPoolExecutor(max_workers=4) as pool:
        for group in pool.map(worker,[selected[i::4] for i in range(4)]): pending['images'].update(group)
    (root/'manifest.json').write_text(json.dumps(pending,indent=2),encoding='utf-8')
    return pending

def load_manifest(root):
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    ids=manifest['ids']
    if len(set(ids))!=len(ids) or not all(re.fullmatch(r'\d{6}',key) for key in ids):
        raise ValueError('Invalid sample IDs')
    for key in ids:
        if sha256((root/'images'/f'{key}.png').read_bytes())!=manifest['images'][key]:
            raise ValueError(f'Image checksum mismatch: {key}')
    return manifest
