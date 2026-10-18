import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
import cv2
import numpy as np
from pedestrian_lab.server import make_handler

class FakeDetector:
    def detect(self,image):return [{'box':[30,20,60,90],'score':.9}],12

class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(detector=FakeDetector()))
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        cls.port=cls.server.server_port
    @classmethod
    def tearDownClass(cls):cls.server.shutdown();cls.server.server_close();cls.thread.join()
    def request(self,path='/',method='GET',body=None,headers=None):
        conn=http.client.HTTPConnection('127.0.0.1',self.port,timeout=5)
        conn.request(method,path,body=body,headers=headers or {});response=conn.getresponse();payload=response.read();status=response.status
        conn.close();return status,payload
    def headers(self):return {'Origin':f'http://127.0.0.1:{self.port}','Content-Type':'image/png'}
    def test_rebinding_host_rejected(self):self.assertEqual(self.request(headers={'Host':'attacker.test'})[0],403)
    def test_cross_origin_rejected(self):
        h=self.headers();h['Origin']='https://attacker.test';self.assertEqual(self.request('/api/detect','POST',b'x',h)[0],403)
    def test_missing_origin_rejected(self):self.assertEqual(self.request('/api/detect','POST',b'x',{'Content-Type':'image/png'})[0],403)
    def test_invalid_media_type_rejected(self):
        h=self.headers();h['Content-Type']='text/plain';self.assertEqual(self.request('/api/detect','POST',b'x',h)[0],415)
    def test_oversized_body_rejected_before_reading(self):
        h=self.headers();h['Content-Length']='8000001';self.assertEqual(self.request('/api/detect','POST',b'x',h)[0],413)
    def test_invalid_image_rejected(self):self.assertEqual(self.request('/api/detect','POST',b'not an image',self.headers())[0],400)
    def test_local_image_processed(self):
        ok,buf=cv2.imencode('.png',np.zeros((100,100,3),np.uint8));self.assertTrue(ok)
        status,payload=self.request('/api/detect','POST',buf.tobytes(),self.headers())
        self.assertEqual(status,200);self.assertEqual(json.loads(payload)['state'],'PRIORITY')
    def test_private_and_parent_paths_not_served(self):
        for path in ('/.git/config','/reports/images/../../../pyproject.toml','/reports/images/../../README.md','/reports/images/../../web/app.js','/models/yolox_s.onnx'):
            self.assertEqual(self.request(path)[0],404)

if __name__=='__main__':unittest.main()
