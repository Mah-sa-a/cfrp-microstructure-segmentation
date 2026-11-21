"""Versioned synthetic three-phase image generation; NumPy and Pillow only."""
import argparse, csv, hashlib, json, os, platform, shutil, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import PIL

def field(rng, size, cells):
    a=rng.uniform(0,255,(cells,cells)).astype('uint8')
    a=np.asarray(Image.fromarray(a).resize((size,size),Image.Resampling.BICUBIC),dtype=np.float32)
    return (a-a.mean())/(a.std()+1e-6)

def make_one(task):
    idx,cfg,directory=task
    directory=Path(directory)
    gs=np.random.SeedSequence([cfg['seed'],idx,0])
    rs=np.random.SeedSequence([cfg['seed'],idx,1])
    g=np.random.default_rng(gs);r=np.random.default_rng(rs)
    size=cfg['parent_size'];pad=cfg['padding'];n=size+2*pad
    scale=cfg['scales'][idx%len(cfg['scales'])]
    d=float(g.uniform(*scale['mean_diameter_px']))
    channel=field(g,n,int(g.integers(4,9)))
    threshold=float(g.uniform(-1.6,-0.3))
    void=np.zeros((n,n),bool)
    target=0. if g.random()<cfg['void_free_probability'] else float(g.uniform(*cfg['void_fraction_range']))
    target_px=target*size*size; area=0; count=0
    center=(slice(pad,pad+size),slice(pad,pad+size))
    for attempt in range(cfg['max_void_attempts']):
        if area>=target_px: break
        cx,cy=g.uniform(0,n,2)
        ax=g.uniform(*cfg['void_diameter_in_fiber_units'])*d/2
        by=ax*g.uniform(.25,1.)
        ang=g.uniform(0,np.pi)
        t=np.linspace(0,2*np.pi,64,endpoint=False)
        phase=g.uniform(0,2*np.pi,3)
        radial=1+.13*np.sin(3*t+phase[0])+.09*np.sin(5*t+phase[1])+.05*np.sin(9*t+phase[2])
        x=ax*radial*np.cos(t);y=by*radial*np.sin(t)
        coords=list(zip(cx+x*np.cos(ang)-y*np.sin(ang),cy+x*np.sin(ang)+y*np.cos(ang)))
        canvas=Image.new('L',(n,n),0);ImageDraw.Draw(canvas).polygon(coords,fill=1)
        candidate=np.asarray(canvas,dtype=bool)
        addition=int(np.count_nonzero(candidate[center]&~void[center]))
        if not addition: continue
        if area+addition>target_px+cfg['void_fraction_tolerance']*size*size: continue
        void|=candidate;area+=addition;count+=1
    if target and abs(area/(size*size)-target)>cfg['void_fraction_tolerance']:
        raise RuntimeError(f'Parent {idx}: void target not achieved within bounded attempts')
    mask=np.zeros((n,n),np.uint8);mask[void]=2
    intens=np.zeros((n,n),np.float32)
    fiber_mean=float(r.uniform(*cfg['fiber_mean_range']))
    matrix_mean=float(r.uniform(*cfg['matrix_mean_range']))
    void_mean=float(r.uniform(*cfg['void_mean_range']))
    # Spatial bins enforce a circumscribed-circle exclusion, including ellipses.
    cell=d*1.5;bins={};fibers=[]
    for attempt in range(cfg['fiber_attempts_per_parent']):
        x,y=g.uniform(0,n,2)
        if channel[min(int(y),n-1),min(int(x),n-1)]<threshold: continue
        diameter=float(np.clip(g.normal(d,.10*d),.72*d,1.28*d))
        a=diameter/2;b=a*g.uniform(.83,1.);angle=g.uniform(0,np.pi)
        bx,by=int(x//cell),int(y//cell)
        reject=False
        for iy in range(by-1,by+2):
            for ix in range(bx-1,bx+2):
                for xx,yy,aa in bins.get((ix,iy),[]):
                    if (x-xx)**2+(y-yy)**2<(a+aa+.25)**2:
                        reject=True;break
                if reject: break
            if reject: break
        if reject: continue
        x0=max(0,int(np.floor(x-a-1)));x1=min(n,int(np.ceil(x+a+1)))
        y0=max(0,int(np.floor(y-a-1)));y1=min(n,int(np.ceil(y+a+1)))
        yy,xx=np.mgrid[y0:y1,x0:x1];dx=xx-x;dy=yy-y
        local=((dx*np.cos(angle)+dy*np.sin(angle))/a)**2+((-dx*np.sin(angle)+dy*np.cos(angle))/b)**2<=1
        region=mask[y0:y1,x0:x1]
        if not local.any() or np.any(region[local]!=0): continue
        region[local]=1
        intens[y0:y1,x0:x1][local]=r.normal(0,7)
        bins.setdefault((bx,by),[]).append((x,y,a))
        fibers.append([float(x-pad),float(y-pad),float(a),float(b),float(angle)])
    mask=mask[center].copy();intens=intens[center]
    if not np.any(mask==1) or not np.any(mask==0):
        raise RuntimeError(f'Parent {idx}: degenerate matrix/fiber geometry')
    means=np.array([matrix_mean,fiber_mean,void_mean],np.float32)
    image=means[mask]+intens
    texture=field(r,size,64);illum=field(r,size,4)
    image+=texture*r.uniform(1,5)+illum*r.uniform(1,9)
    noise_std=float(r.uniform(.5,3.5));blur=float(r.uniform(.15,.85))
    image=np.asarray(Image.fromarray(np.clip(image,0,255).astype('uint8')).filter(ImageFilter.GaussianBlur(blur)),dtype=np.float32)
    image=np.clip(image+r.normal(0,noise_std,image.shape),0,255).astype('uint8')
    uid=f'syn_{idx:07d}';ip=directory/'images'/f'{uid}.png';mp=directory/'masks'/f'{uid}.png'
    Image.fromarray(image).save(ip,compress_level=2)
    Image.fromarray(mask).save(mp,compress_level=2)
    for p,expected in [(ip,image),(mp,mask)]:
        with Image.open(p) as im:
            if im.mode!='L' or not np.array_equal(np.asarray(im),expected):
                raise RuntimeError(f'PNG round-trip failed: {p}')
    fractions=np.bincount(mask.ravel(),minlength=3)/(size*size)
    meta=dict(parent_id=uid,geometry_seed_components=[cfg['seed'],idx,0],render_seed_components=[cfg['seed'],idx,1],
        scale_family=scale['name'],mean_diameter_parameter_px=d,channel_threshold=threshold,
        target_void_fraction=target,achieved_phase_fractions=fractions.tolist(),void_shapes_added=count,
        accepted_fibers_in_padded_field=len(fibers),fiber_attempts=cfg['fiber_attempts_per_parent'],
        fiber_mean=fiber_mean,matrix_mean=matrix_mean,void_mean=void_mean,blur_sigma_px=blur,noise_std=noise_std,
        fibers= fibers,uncertain_bright_islands=False)
    (directory/'metadata'/f'{uid}.json').write_text(json.dumps(meta,indent=2))
    if idx<cfg['qc_parents']:
        pal=np.array([[100,100,100],[240,190,80],[40,100,200]],np.uint8)
        panel=Image.new('RGB',(size*2,size));panel.paste(Image.fromarray(image).convert('RGB'),(0,0));panel.paste(Image.fromarray(pal[mask]),(size,0))
        panel.thumbnail((1024,512));panel.save(directory/'qc'/f'{uid}.jpg',quality=90)
    return dict(parent_id=uid,split_group_id=uid,image_path=str(ip),mask_path=str(mp),scale_family=scale['name'],
        matrix_fraction=float(fractions[0]),fiber_fraction=float(fractions[1]),void_fraction=float(fractions[2]),
        image_sha256=hashlib.sha256(ip.read_bytes()).hexdigest(),mask_sha256=hashlib.sha256(mp.read_bytes()).hexdigest())

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text());dest=Path(a.out).resolve()
    if cfg['parent_size']%512: raise ValueError('Parent size must be a multiple of 512')
    if cfg['parents']<3: raise ValueError('Need at least three parents')
    dest.mkdir(parents=True,exist_ok=False)
    for name in ['images','masks','metadata','qc']: (dest/name).mkdir()
    shutil.copy2(a.config,dest/'generation_config.json');shutil.copy2(__file__,dest/'generator_source.py')
    prov=dict(python=sys.version,numpy=np.__version__,pillow=PIL.__version__,platform=platform.platform(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        parameter_basis='Provisional design ranges, not fitted measurements; real-image qualitative design guidance',
        note='No training is launched. QC previews are not manual approval. Geometry is not a manufacturing simulation.')
    (dest/'provenance.json').write_text(json.dumps(prov,indent=2))
    start=time.time();rows=[]
    with (dest/'pairs.partial.csv').open('w',newline='') as f:
        writer=None
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for row in pool.map(make_one,((i,cfg,str(dest)) for i in range(cfg['parents'])),chunksize=1):
                if writer is None: writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader()
                writer.writerow(row);f.flush();rows.append(row)
                if len(rows)%25==0: print(json.dumps(dict(completed=len(rows),total=cfg['parents'],seconds=time.time()-start)),flush=True)
    os.replace(dest/'pairs.partial.csv',dest/'pairs.csv')
    summary={}
    for scale in cfg['scales']:
        selected=[x for x in rows if x['scale_family']==scale['name']]
        summary[scale['name']]={k:dict(min=float(min(x[k] for x in selected)),mean=float(np.mean([x[k] for x in selected])),max=float(max(x[k] for x in selected))) for k in ['matrix_fraction','fiber_fraction','void_fraction']}
    (dest/'COMPLETE.json').write_text(json.dumps(dict(parents=len(rows),seconds=time.time()-start,summary=summary),indent=2))
    print('COMPLETE:',dest,flush=True)
if __name__=='__main__': main()
