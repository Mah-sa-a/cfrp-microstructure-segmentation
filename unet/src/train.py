import argparse,csv,json,os,random,time,hashlib,platform
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image
from dataset import ROOT,path,read_rows,PatchDataset
from model import UNet
from metrics import confusion,scores,extra_metrics
from validate_dataset import validate

def clean(value):
    if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
    if isinstance(value,list): return [clean(v) for v in value]
    if isinstance(value,(np.integer,)): return int(value)
    if isinstance(value,(float,np.floating)) and not np.isfinite(value): return None
    return value

def save_json(filename,value):
    filename.write_text(json.dumps(clean(value),indent=2),encoding='utf-8')

def save_csv(filename,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with filename.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def checkpoint(filename,payload):
    temporary=filename.with_suffix('.tmp');torch.save(payload,temporary);os.replace(temporary,filename)

def seed_worker(worker_id):
    seed=torch.initial_seed()%2**32;random.seed(seed);np.random.seed(seed)

def loader(rows,cfg,train=False,generator=None):
    return DataLoader(PatchDataset(rows,augment=train),batch_size=cfg['batch_size'],shuffle=train,
                      num_workers=cfg['workers'],pin_memory=True,worker_init_fn=seed_worker,
                      generator=generator,drop_last=False)

def loss_fn(logits,target):
    ce=nn.functional.cross_entropy(logits,target)
    probs=logits.softmax(1)
    onehot=nn.functional.one_hot(target,3).permute(0,3,1,2).float()
    inter=(probs*onehot).sum((0,2,3));den=(probs+onehot).sum((0,2,3))
    dice=1-((2*inter+1e-6)/(den+1e-6)).mean()
    return .5*ce+.5*dice

@torch.no_grad()
def evaluate(model,data,rows,device,cfg,out=None):
    model.eval(); groups=defaultdict(lambda:np.zeros((3,3),dtype=np.int64));records=[]
    total_loss=0.;n=0;total_cm=np.zeros((3,3),dtype=np.int64)
    if out: (out/'predictions').mkdir(exist_ok=True)
    for image,target,indices in data:
        image=image.to(device,non_blocking=True);target=target.to(device,non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type=='cuda' and cfg.get('precision','amp_fp16')=='amp_fp16'):
            logits=model(image);loss=loss_fn(logits,target)
        probs=logits.float().softmax(1).cpu().numpy();pred=probs.argmax(1);truth=target.cpu().numpy()
        total_loss+=loss.item()*len(image);n+=len(image)
        for j,index in enumerate(indices.tolist()):
            row=rows[index];cm=confusion(truth[j],pred[j]);total_cm+=cm
            group=row['split_group_id'];groups[group]+=cm
            if out:
                record=dict(dataset_sample_id=row['dataset_sample_id'],source_folder=row.get('source_folder','synthetic'),
                            split_group_id=group,parent_id=row.get('parent_id',''),**scores(cm),
                            **extra_metrics(truth[j],pred[j],cfg['boundary_tolerance_px']))
                onehot=np.eye(3)[truth[j]].transpose(2,0,1)
                record['multiclass_brier']=float(((probs[j]-onehot)**2).sum(0).mean())
                records.append(record)
                if cfg['save_all_predictions']:
                    Image.fromarray(pred[j].astype('uint8')).save(out/'predictions'/f'{index:06d}.png')
    group_records=[dict(split_group_id=g,**scores(cm)) for g,cm in sorted(groups.items())]
    result=dict(loss=total_loss/max(n,1),patches=n,pooled=scores(total_cm),
                group_macro_iou=float(np.mean([r['macro_iou'] for r in group_records])),
                confusion_matrix=total_cm.tolist(),groups=group_records)
    if out:
        save_csv(out/'test_metrics_per_patch.csv',records);save_csv(out/'test_metrics_per_group.csv',group_records)
        save_csv(out/'prediction_index.csv',[dict(prediction=f'predictions/{i:06d}.png',**rows[i]) for i in range(len(rows))])
        np.savetxt(out/'confusion_matrix.csv',total_cm,delimiter=',',fmt='%d')
        np.savetxt(out/'confusion_matrix_reference_normalized.csv',np.divide(total_cm,total_cm.sum(1,keepdims=True),
                   out=np.full((3,3),np.nan),where=total_cm.sum(1,keepdims=True)>0),delimiter=',')
        save_json(out/'test_summary.json',result)
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',required=True,choices=['real_only','synthetic_pretrain','synthetic_to_real','synthetic_only'])
    parser.add_argument('--fold',type=int,default=1);parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--config',default='unet/configs/training.json');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();cfg=json.loads(path(args.config).read_text());seed=args.seed
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable. Check GPU allocation and PyTorch installation.')
    # Configurable numerical precision.
    precision=cfg.get('precision','amp_fp16')
    if precision not in ('amp_fp16','float32'):
        raise ValueError(f'Unsupported precision: {precision}')
    print('Numerical precision:',precision,flush=True)
    if precision=='float32':
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda');random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    generator=torch.Generator().manual_seed(seed)
    synthetic=args.protocol=='synthetic_pretrain'
    directory=path(cfg['synthetic_splits']) if synthetic else path(cfg['real_splits'])/f'fold_{args.fold:02d}'
    manifests={s:directory/(s+'.csv') for s in ['train','validation','test']}
    print('Validating decoded image/mask pairs before full training...',flush=True)
    report=validate(list(manifests.values()),allow_unreviewed=cfg['allow_unreviewed'] and not synthetic)
    print('Dataset validation passed.',flush=True)
    rows={s:read_rows(filename) for s,filename in manifests.items()}
    stage='pretrain' if synthetic else f'fold_{args.fold:02d}'
    out=ROOT/'outputs/unet_experiments'/cfg['run_name']/args.protocol/f'seed_{seed}'/stage
    ckpt_dir=ROOT/'checkpoints/unet'/cfg['run_name']/args.protocol/f'seed_{seed}'/stage
    if out.exists() and not args.resume: raise FileExistsError(f'{out} already exists. Use --resume or a new run_name.')
    out.mkdir(parents=True,exist_ok=True);ckpt_dir.mkdir(parents=True,exist_ok=True)
    signature=dict(config=cfg,protocol=args.protocol,seed=seed,fold=args.fold,
                   manifest_hashes={s:hashlib.sha256(p.read_bytes()).hexdigest() for s,p in manifests.items()})
    signature_file=out/'run_signature.json'
    if signature_file.exists() and json.loads(signature_file.read_text())!=signature:
        raise ValueError('Configuration/manifests changed; use a new run_name.')
    save_json(signature_file,signature);save_json(out/'dataset_audit.json',report)
    save_json(out/'environment.json',dict(python=platform.python_version(),torch=torch.__version__,
              cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),label_status='exploratory' if cfg['allow_unreviewed'] and not synthetic else 'reviewed_policy'))
    save_json(out/'code_provenance.json',dict(
        precision=precision,
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        tf32_cudnn=torch.backends.cudnn.allow_tf32,
        source_sha256={
            name:hashlib.sha256((ROOT/'unet/src'/name).read_bytes()).hexdigest()
            for name in ['train.py','model.py','dataset.py','metrics.py','validate_dataset.py']
        }
    ))
    model=UNet(cfg['base_channels']).to(device)
    initial=ROOT/'checkpoints/unet'/cfg['run_name']/'synthetic_pretrain'/f'seed_{seed}'/'pretrain'/'best.pt'
    if args.protocol in ('synthetic_to_real','synthetic_only'):
        if not initial.is_file(): raise FileNotFoundError(f'Pretraining checkpoint missing: {initial}')
        model.load_state_dict(torch.load(initial,map_location='cpu',weights_only=False)['model'])
        save_json(out/'initialization.json',dict(checkpoint=str(initial.relative_to(ROOT)),sha256=hashlib.sha256(initial.read_bytes()).hexdigest()))
    evaluation_loader=loader(rows['test'],cfg)
    if args.protocol=='synthetic_only':
        evaluate(model,evaluation_loader,rows['test'],device,cfg,out);return
    train_loader=loader(rows['train'],cfg,True,generator);val_loader=loader(rows['validation'],cfg)
    lr=cfg['finetune_lr'] if args.protocol=='synthetic_to_real' else cfg['lr']
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=5)
    scaler=torch.cuda.amp.GradScaler(enabled=precision=='amp_fp16');start=0;best=-float('inf');bad=0;history=[]
    last=ckpt_dir/'last.pt'
    if args.resume and last.is_file():
        state=torch.load(last,map_location='cpu',weights_only=False);model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
        start=state['epoch']+1;best=state['best'];bad=state['bad'];history=state['history']
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng']);generator.set_state(state['loader_rng'])
    torch.cuda.reset_peak_memory_stats();wall_start=time.time()
    for epoch in range(start,cfg['epochs']):
        if bad>=cfg['patience']: break
        model.train();total=0.;n=0;epoch_start=time.time()
        for batch_number,(image,target,indices) in enumerate(train_loader):
            image=image.to(device,non_blocking=True);target=target.to(device,non_blocking=True);optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=precision=='amp_fp16'):
                logits=model(image)
                loss=loss_fn(logits,target)
            if not torch.isfinite(loss):
                def tensor_report(tensor):
                    detached=tensor.detach()
                    finite=torch.isfinite(detached)
                    values=detached[finite]
                    return dict(
                        dtype=str(detached.dtype),
                        shape=list(detached.shape),
                        all_finite=bool(finite.all().item()),
                        nonfinite_count=int((~finite).sum().item()),
                        finite_min=float(values.min().item()) if values.numel() else None,
                        finite_max=float(values.max().item()) if values.numel() else None
                    )
                diagnostic=dict(
                    epoch=epoch+1,
                    batch_number=batch_number,
                    seed=seed,
                    fold=args.fold,
                    loss=tensor_report(loss),
                    image=tensor_report(image),
                    logits=tensor_report(logits),
                    target_classes=target.detach().unique().cpu().tolist(),
                    samples=[rows['train'][int(i)] for i in indices.tolist()]
                )
                diagnostic['per_sample']=[
                    dict(
                        dataset_sample_id=rows['train'][int(index)].get('dataset_sample_id',''),
                        image=tensor_report(image[j]),
                        logits=tensor_report(logits[j])
                    )
                    for j,index in enumerate(indices.tolist())
                ]
                diagnostic['nonfinite_parameter_names']=[
                    name for name,parameter in model.named_parameters()
                    if not torch.isfinite(parameter.detach()).all().item()
                ]
                try:
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=False):
                            fp32_logits=model(image.float())
                            fp32_loss=loss_fn(fp32_logits,target)
                    diagnostic['fp32_comparison']=dict(
                        logits=tensor_report(fp32_logits),
                        loss=tensor_report(fp32_loss),
                        per_sample_logits=[
                            tensor_report(fp32_logits[j])
                            for j in range(len(image))
                        ]
                    )
                except Exception as error:
                    diagnostic['fp32_comparison']=dict(
                        error_type=type(error).__name__,
                        error=str(error)
                    )
                diagnostic_file=out/f'nonfinite_epoch_{epoch+1:03d}_batch_{batch_number:05d}.json'
                save_json(diagnostic_file,diagnostic)
                print('NONFINITE DIAGNOSTIC: '+json.dumps(diagnostic),flush=True)
                raise RuntimeError(f'Non-finite training loss; diagnostic saved to {diagnostic_file}')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            scaler.step(optimizer);scaler.update();total+=loss.item()*len(image);n+=len(image)
        val=evaluate(model,val_loader,rows['validation'],device,cfg);metric=val['group_macro_iou'];scheduler.step(metric)
        improved=metric>best
        if improved: best=metric;bad=0
        else: bad+=1
        record=dict(epoch=epoch+1,train_loss=total/max(n,1),validation_loss=val['loss'],validation_group_macro_iou=metric,
                    lr=optimizer.param_groups[0]['lr'],seconds=time.time()-epoch_start)
        history.append(record);save_csv(out/'training_history.csv',history)
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
                   epoch=epoch,best=best,bad=bad,history=history,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                   torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),loader_rng=generator.get_state(),config=cfg)
        checkpoint(last,state)
        if improved: checkpoint(ckpt_dir/'best.pt',dict(model=model.state_dict(),epoch=epoch,config=cfg))
        print(json.dumps(record),flush=True)
    model.load_state_dict(torch.load(ckpt_dir/'best.pt',map_location='cpu',weights_only=False)['model'])
    torch.cuda.synchronize();t=time.time();test=evaluate(model,evaluation_loader,rows['test'],device,cfg,out);torch.cuda.synchronize()
    save_json(out/'timing.json',dict(current_job_seconds=time.time()-wall_start,test_total_seconds=time.time()-t,
              peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),parameters=sum(p.numel() for p in model.parameters())))
    print('Completed:',out,flush=True)

if __name__=='__main__': main()
