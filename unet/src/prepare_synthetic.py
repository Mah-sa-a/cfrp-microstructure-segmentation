"""Input: manually verified image/mask pair CSV with geometry-level split_group_id."""
import argparse,csv,json,random,math
from PIL import Image
from dataset import ROOT,path

def positions(length):
    if length<512: raise ValueError('Synthetic images must be at least 512x512')
    return sorted(set(list(range(0,length-511,512))+[length-512]))

def main():
    p=argparse.ArgumentParser();p.add_argument('--pairs',required=True);p.add_argument('--out',default='data/splits/synthetic');p.add_argument('--seed',type=int,default=2026)
    a=p.parse_args()
    with path(a.pairs).open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        if not {'image_path','mask_path','split_group_id'}.issubset(reader.fieldnames or []):
            raise ValueError('Need image_path, mask_path, split_group_id columns')
        pairs=list(reader)
    groups=sorted({r['split_group_id'].strip() for r in pairs})
    if not groups or '' in groups or len(groups)<3: raise ValueError('Need at least three nonempty independent geometry groups')
    random.Random(a.seed).shuffle(groups)
    n=max(1,int(math.floor(.15*len(groups))));test=set(groups[:n]);val=set(groups[n:2*n]);train=set(groups[2*n:])
    destination=path(a.out)
    if destination.exists(): raise FileExistsError(destination)
    partitions={s:[] for s in ['train','validation','test']};seen=set()
    for i,row in enumerate(pairs):
        image=path(row['image_path']);mask=path(row['mask_path']);key=str(image.resolve())
        if key in seen: raise ValueError('Repeated image path '+key)
        seen.add(key)
        with Image.open(image) as im: w,h=im.size
        with Image.open(mask) as im:
            if im.size!=(w,h): raise ValueError('Image/mask size mismatch')
        group=row['split_group_id'].strip();partition='test' if group in test else 'validation' if group in val else 'train'
        for y in positions(h):
            for x in positions(w):
                uid=f'synthetic_{i:07d}_y{y}_x{x}'
                partitions[partition].append(dict(image_path=str(image.relative_to(ROOT)),mask_path=str(mask.relative_to(ROOT)),
                    split_group_id=group,parent_id=group,dataset_sample_id=uid,sample_id=uid,source_folder='synthetic',
                    quality_flag='good',crop_x=x,crop_y=y))
    destination.mkdir(parents=True)
    for s,rows in partitions.items():
        with (destination/(s+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (destination/'group_assignment.json').write_text(json.dumps(dict(train=sorted(train),validation=sorted(val),test=sorted(test)),indent=2))
    print({s:len(r) for s,r in partitions.items()})
    print('Raw label IDs MUST be matrix=0, fiber=1, void=2. They are validated before training; no remapping is guessed.')

if __name__=='__main__':main()
