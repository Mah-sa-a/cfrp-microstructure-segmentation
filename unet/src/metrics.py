import numpy as np
from scipy import ndimage as ndi
CLASSES=('matrix','fiber','void')

def confusion(target,pred):
    return np.bincount(3*target.ravel().astype(int)+pred.ravel().astype(int),minlength=9).reshape(3,3)

def scores(cm):
    tp=np.diag(cm).astype(float); ref=cm.sum(1); pred=cm.sum(0); union=ref+pred-tp
    divide=lambda a,b: np.divide(a,b,out=np.full(3,np.nan),where=b>0)
    result={}
    for c,v in enumerate(CLASSES):
        for key,vals in [('iou',divide(tp,union)),('dice',divide(2*tp,ref+pred)),
                         ('precision',divide(tp,pred)),('recall',divide(tp,ref))]:
            result[v+'_'+key]=float(vals[c])
    result['macro_iou']=float(np.nanmean(divide(tp,union)))
    result['pixel_accuracy']=float(tp.sum()/max(cm.sum(),1))
    return result

def boundary(mask):
    return mask & ~ndi.binary_erosion(mask,structure=ndi.generate_binary_structure(2,1),border_value=0)

def extra_metrics(target,pred,tolerance=2):
    result={}
    for c,name in enumerate(CLASSES):
        a=target==c; b=pred==c
        result[name+'_reference_fraction']=float(a.mean())
        result[name+'_predicted_fraction']=float(b.mean())
        result[name+'_fraction_bias_pp']=float(100*(b.mean()-a.mean()))
        result[name+'_fraction_abs_error_pp']=abs(result[name+'_fraction_bias_pp'])
        result[name+'_absent_reference']=int(not a.any())
        result[name+'_false_positive_pixels']=int(b.sum()) if not a.any() else 0
        ba,bb=boundary(a),boundary(b)
        if ba.any() and bb.any():
            da=ndi.distance_transform_edt(~ba)[bb]; db=ndi.distance_transform_edt(~bb)[ba]
            precision=float((da<=tolerance).mean()); recall=float((db<=tolerance).mean())
            result[name+'_boundary_f1']=2*precision*recall/(precision+recall) if precision+recall else 0.
            result[name+'_assd_px']=float((da.sum()+db.sum())/(len(da)+len(db)))
        else:
            result[name+'_boundary_f1']=float('nan') if not ba.any() and not bb.any() else 0.
            result[name+'_assd_px']=float('nan')
    for label,m in [('reference',target==2),('predicted',pred==2)]:
        objects,n=ndi.label(m,structure=np.ones((3,3)))
        areas=np.bincount(objects.ravel())[1:]
        result['void_'+label+'_components']=int(n)
        result['void_'+label+'_largest_area_px']=int(areas.max()) if n else 0
        result['void_'+label+'_mean_component_area_px']=float(areas.mean()) if n else 0.
    return result
