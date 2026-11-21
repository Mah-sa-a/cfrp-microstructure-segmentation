import argparse,json
import numpy as np
from PIL import Image
import torch
from model import UNet
from dataset import path

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--image',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--overlap',type=int,default=128)
    a=p.parse_args()
    if not 0<=a.overlap<512:raise ValueError('Overlap must be 0..511')
    output=path(a.output)
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    with Image.open(path(a.image)) as im:
        if im.mode.startswith('I') or im.mode=='F':raise ValueError('Explicit high-bit-depth normalization required')
        image=np.array(im.convert('L'),dtype=np.uint8)
    h,w=image.shape;canvas=np.pad(image,((0,max(0,512-h)),(0,max(0,512-w))),mode='edge');hh,ww=canvas.shape
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state=torch.load(path(a.checkpoint),map_location='cpu',weights_only=False)
    model=UNet(state['config']['base_channels']).to(device);model.load_state_dict(state['model']);model.eval()
    probs=np.zeros((3,hh,ww),dtype=np.float32);count=np.zeros((hh,ww),dtype=np.float32)
    pos=lambda n: sorted(set(list(range(0,n-511,512-a.overlap))+[n-512]))
    for y in pos(hh):
        for x in pos(ww):
            tile=canvas[y:y+512,x:x+512].astype(np.float32)/255.
            tensor=torch.from_numpy(tile[None,None]).to(device)
            probs[:,y:y+512,x:x+512]+=model(tensor).softmax(1)[0].cpu().numpy()
            count[y:y+512,x:x+512]+=1
    pred=(probs/count[None]).argmax(0)[:h,:w].astype('uint8')
    Image.fromarray(pred).save(output/'mask_raw.png')
    colors=np.array([[128,128,128],[255,0,0],[0,255,0]],dtype=np.uint8)
    Image.fromarray(colors[pred]).save(output/'mask_color.png')
    rgb=np.repeat(image[:,:,None],3,axis=2)
    Image.fromarray((.6*rgb+.4*colors[pred]).astype('uint8')).save(output/'overlay.png')
    (output/'phase_area_fractions.json').write_text(json.dumps({name:float((pred==c).mean()) for c,name in enumerate(['matrix','fiber','void'])},indent=2))
    print(output)

if __name__=='__main__':main()
