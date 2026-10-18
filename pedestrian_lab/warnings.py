"""An explicit image-space attention heuristic; not range estimation or collision prediction."""
import math

def attention(detections,width,height):
    if width<=0 or height<=0: raise ValueError('Invalid image dimensions')
    output=[]
    for d in detections:
        b=d['box']
        if len(b)!=4 or not all(math.isfinite(v) for v in b) or b[2]<=b[0] or b[3]<=b[1]:
            raise ValueError('Invalid detection box')
        x=(b[0]+b[2])/(2*width);y=b[3]/height
        half=.08+.27*max(0,min(1,(y-.45)/.55))
        in_corridor=.45<=y<=1 and abs(x-.5)<=half
        level='PRIORITY' if in_corridor and y>=.72 else 'WATCH' if in_corridor else 'PERIPHERAL'
        output.append({**d,'attention':level,'foot':[x,y]})
    level='PRIORITY' if any(d['attention']=='PRIORITY' for d in output) else 'WATCH' if any(d['attention']=='WATCH' for d in output) else 'DETECTIONS' if output else 'NO_CANDIDATES'
    return {'state':level,'detections':output,'basis':'Bounding-box bottom center in a fixed image-space trapezoid; no lane or depth estimation'}
