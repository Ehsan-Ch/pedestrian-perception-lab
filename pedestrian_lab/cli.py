import argparse,json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description='Pedestrian Perception Lab')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--data',default='data/kitti');p.add_argument('--per-group',type=int,default=32)
    p=sub.add_parser('evaluate');p.add_argument('--data',default='data/kitti');p.add_argument('--model',default='models/yolox_s.onnx');p.add_argument('--output',default='reports/custom')
    p=sub.add_parser('serve');p.add_argument('--port',type=int,default=8783);p.add_argument('--model',default='models/yolox_s.onnx')
    p=sub.add_parser('detect');p.add_argument('image');p.add_argument('--model',default='models/yolox_s.onnx');p.add_argument('--output',default='detections.json')
    args=parser.parse_args()
    if args.command=='prepare':
        from .data import prepare
        prepare(args.data,args.per_group)
    elif args.command=='evaluate':
        from .evaluate import evaluate
        result=evaluate(args.data,args.model,args.output);print(json.dumps(result['summary'],indent=2))
    elif args.command=='serve':
        from .server import serve
        serve(args.port,args.model)
    else:
        import cv2
        from .detector import Detector
        from .warnings import attention
        image=cv2.imread(args.image);detections,ms=Detector(args.model).detect(image)
        result={**attention(detections,image.shape[1],image.shape[0]),'latency_ms':ms}
        output=Path(args.output)
        with output.open('x',encoding='utf-8') as f: json.dump(result,f,indent=2,allow_nan=False)

if __name__=='__main__': main()
