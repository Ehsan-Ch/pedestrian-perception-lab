import json
import platform
import time
from pathlib import Path
import cv2
import numpy as np
from .data import load_manifest,sha256
from .detector import Detector,HogBaseline,CONFIDENCE,NMS,MODEL_SHA256
from .metrics import match,aggregate,subgroup_recall
from .warnings import attention

def evaluate(data,model,output):
    data,output=Path(data),Path(output)
    if (output/'results.json').exists(): raise ValueError('Results already exist; use a fresh output directory')
    output.mkdir(parents=True,exist_ok=True);(output/'images').mkdir(exist_ok=True)
    manifest=load_manifest(data);cv2.setNumThreads(2)
    detector=Detector(model);hog=HogBaseline()
    detector.detect(np.zeros((384,1248,3),np.uint8)) # Exclude one warm-up from latency.
    variants={'yolox':[],'dim':[],'hog':[]};latencies={k:[] for k in variants};scenes=[]
    for key in manifest['ids']:
        image=cv2.imread(str(data/'images'/f'{key}.png'))
        if image is None: raise ValueError(f'Cannot decode {key}')
        dim=np.rint(image.astype(float)*.35).astype(np.uint8)
        record={'id':key,'width':image.shape[1],'height':image.shape[0],'image':f'images/{key}.jpg','annotations':manifest['annotations'][key],'variants':{}}
        for name,engine,pixels in [('yolox',detector,image),('dim',detector,dim),('hog',hog,image)]:
            det,ms=engine.detect(pixels);scored=match(det,manifest['annotations'][key])
            scored['latency_ms']=round(ms,3);scored['attention']=attention(det,image.shape[1],image.shape[0])
            record['variants'][name]=scored;variants[name].append(scored);latencies[name].append(ms)
        cv2.imwrite(str(output/'images'/f'{key}.jpg'),image,[cv2.IMWRITE_JPEG_QUALITY,86])
        record['preview_sha256']=sha256((output/record['image']).read_bytes())
        scenes.append(record);print(f'Evaluated {key}: {record["variants"]["yolox"]["tp"]} matched, {record["variants"]["yolox"]["fn"]} missed',flush=True)
    annotations=[manifest['annotations'][k] for k in manifest['ids']]
    results={'protocol':{'sample_size':len(scenes),'seed':manifest['seed'],'selection':manifest['selection'],
              'pretrained':'YOLOX-S COCO, OpenCV Zoo; no fine-tuning or threshold tuning on this sample',
              'confidence':CONFIDENCE,'nms_iou':NMS,'matching_iou':.5,'scored_class':'All Pedestrian boxes, including small/truncated/occluded',
              'ignored':'Unmatched detections overlapping Cyclist/Person_sitting by IoU >= 0.5 or DontCare by detection-area overlap > 0.5; duplicates remain false positives',
              'metric':'Fixed-threshold micro precision/recall, not official KITTI AP; equal positive/negative image sampling changes prevalence',
              'dim_stress':'Each BGR intensity multiplied by 0.35 and rounded; synthetic perturbation, not real night-driving data',
              'latency':'Local CPU wall time for preprocessing, inference and postprocessing, 2 OpenCV threads; excludes image I/O and UI; one warm-up excluded'},
             'environment':{'python':platform.python_version(),'opencv':cv2.__version__,'numpy':np.__version__,'platform':platform.platform(),'processor':platform.processor()},
             'model_sha256':MODEL_SHA256,'dataset':{k:v for k,v in manifest.items() if k!='annotations'},'summary':{},'scenes':scenes}
    for name,frames in variants.items():
        results['summary'][name]={**aggregate(frames),'median_ms':float(np.median(latencies[name])),'p95_ms':float(np.percentile(latencies[name],95)),
            'small':subgroup_recall(frames,annotations,lambda a:a['box'][3]-a['box'][1]<40),
            'occluded':subgroup_recall(frames,annotations,lambda a:a['occlusion']>=1)}
    results['source_sha256']={p.name:sha256(p.read_bytes()) for p in Path(__file__).parent.glob('*.py')}
    (output/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
    return results
