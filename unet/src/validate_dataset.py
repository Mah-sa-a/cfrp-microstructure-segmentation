import argparse,json,hashlib
import numpy as np
from dataset import ROOT,path,read_rows,load_pair

def validate(files,allow_unreviewed=False):
    cache={}; partitions=[]; report=[]; hashes={}
    for filename in files:
        rows=read_rows(filename); ids=set(); groups=set(); counts=np.zeros(3,dtype=np.int64)
        for row in rows:
            uid=row.get('dataset_sample_id',row.get('sample_id',''))
            if not uid or uid in ids: raise ValueError('Missing/repeated ID in '+str(filename))
            ids.add(uid)
            group=row.get('split_group_id',row.get('parent_id',''))
            if not group: raise ValueError('Missing group ID')
            groups.add(group)
            flag=row.get('quality_flag','').lower()
            if flag!='good' and not (allow_unreviewed and flag=='unreviewed'):
                raise ValueError('Unapproved review status: '+flag)
            key=(row['image_path'],row['mask_path'],row.get('crop_x'),row.get('crop_y'))
            if key not in cache:
                image,mask=load_pair(row)
                h=hashlib.sha256(image.tobytes()).hexdigest()
                if h in hashes and hashes[h]!=group:
                    raise ValueError('Decoded duplicate image across groups: '+uid)
                hashes[h]=group
                cache[key]=np.bincount(mask.ravel(),minlength=3)
            counts+=cache[key]
        partitions.append((ids,groups))
        report.append(dict(manifest=str(filename),patches=len(rows),groups=sorted(groups),class_pixels=counts.tolist()))
    for i,(ids,groups) in enumerate(partitions):
        for other_ids,other_groups in partitions[i+1:]:
            if ids & other_ids or groups & other_groups: raise ValueError('Partition ID/group overlap')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',default='fold_01');p.add_argument('--config',default='unet/configs/training.json')
    a=p.parse_args();cfg=json.loads(path(a.config).read_text())
    report=validate([path(cfg['real_splits'])/a.fold/(s+'.csv') for s in ['train','validation','test']],cfg['allow_unreviewed'])
    print(json.dumps(report,indent=2))
