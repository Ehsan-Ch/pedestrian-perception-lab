"""Loopback-only image inference with bounded requests; uploaded pixels are never saved."""
import os
os.environ.setdefault('OPENCV_IO_MAX_IMAGE_PIXELS','16777216')
import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
import cv2
import numpy as np
from .detector import Detector
from .warnings import attention

ROOT=Path(__file__).resolve().parents[1]
MAX_UPLOAD=8_000_000

def make_handler(root=ROOT,model=None,detector=None):
    root=Path(root).resolve();lock=threading.Lock();engine=[detector]
    class Handler(BaseHTTPRequestHandler):
        def reply(self,status,body,kind='application/json'):
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)))
            self.send_header('X-Content-Type-Options','nosniff');self.send_header('Cache-Control','no-store')
            self.send_header('Content-Security-Policy',"default-src 'self'; img-src 'self' blob:; connect-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers();self.wfile.write(body)
        def error(self,status,message): self.reply(status,json.dumps({'error':message}).encode())
        def trusted(self,post=False):
            allowed={f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
            host=self.headers.get('Host','')
            if host not in allowed: self.error(403,'Unrecognized local host');return False
            origin=self.headers.get('Origin')
            if (post and origin is None) or (origin is not None and origin!=f'http://{host}'):
                self.error(403,'Same-origin requests only');return False
            return True
        def do_GET(self):
            if not self.trusted(): return
            route=urlsplit(self.path).path
            if route=='/': route='/web/index.html'
            if route not in ('/web/index.html','/web/style.css','/web/app.js','/reports/results.json') and not route.startswith('/reports/images/'):
                self.error(404,'Not found');return
            path=(root/route.lstrip('/')).resolve()
            if route.startswith('/reports/images/') and (path.parent != root/'reports/images' or path.suffix != '.jpg'):
                self.error(404,'Not found');return
            if not path.is_relative_to(root) or not path.is_file(): self.error(404,'Not found');return
            self.reply(200,path.read_bytes(),mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
        def do_POST(self):
            if not self.trusted(post=True): return
            if urlsplit(self.path).path!='/api/detect': self.error(404,'Not found');return
            if self.headers.get('Content-Type','').split(';')[0] not in ('image/jpeg','image/png','image/webp'):
                self.error(415,'Use JPEG, PNG or WebP');return
            try: length=int(self.headers.get('Content-Length','0'))
            except ValueError: self.error(400,'Invalid content length');return
            if not 0<length<=MAX_UPLOAD: self.error(413,'Image must be no larger than 8 MB');return
            if not lock.acquire(blocking=False): self.error(429,'Detector busy; try again');return
            try:
                self.connection.settimeout(15);body=self.rfile.read(length)
                if len(body)!=length: self.error(400,'Incomplete image');return
                image=cv2.imdecode(np.frombuffer(body,np.uint8),cv2.IMREAD_COLOR)
                if image is None: self.error(400,'Cannot decode image');return
                if engine[0] is None: engine[0]=Detector(model or root/'models/yolox_s.onnx')
                detections,ms=engine[0].detect(image)
                result={**attention(detections,image.shape[1],image.shape[0]),'width':image.shape[1],'height':image.shape[0],'latency_ms':round(ms,2)}
                self.reply(200,json.dumps(result,allow_nan=False).encode())
            except FileNotFoundError: self.error(503,'Model missing. Run python scripts/setup.py')
            except (ValueError,cv2.error): self.error(400,'Invalid image or model configuration')
            except TimeoutError: self.error(408,'Image upload timed out')
            finally: lock.release()
        def log_message(self,*args): pass
    return Handler

def serve(port=8783,model=None):
    cv2.setNumThreads(2)
    server=ThreadingHTTPServer(('127.0.0.1',port),make_handler(model=model))
    print(f'Pedestrian Perception Lab: http://127.0.0.1:{port}',flush=True)
    try: server.serve_forever()
    finally: server.server_close()
