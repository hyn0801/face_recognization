#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
import facenet
import align.detect_face
import math
import pickle
from sklearn.svm import SVC
from http.server import BaseHTTPRequestHandler, HTTPServer
from os import path
import urllib
import json
import cgi
# HTTPRequestHandler class

class testHTTPServer_RequestHandler(BaseHTTPRequestHandler):
    # GET
    
    def do_GET(self):
        global a
        print("do_GET",a)
        a+=1
        

    def do_POST(self):
        '''
        print("do_POST",b)
        print(self.path)
        print(self.request_version)
        print(self.requestline)
        print(self.headers['content-type'])
        '''
        if self.headers['content-type']=='application/json':
            global images_placeholder
            global embeddings
            global phase_train_placeholder
            length = int(self.headers['content-length'])
            request = json.loads(self.rfile.read(length))
            print(request["request"])
            print(request["face"])
            stamp = time.time()
            images = align_image(request["face"], args.image_size, args.margin, pnet, rnet, onet)
            elapsed = time.time() - stamp
            print('align spent:  %1.4f  s' % elapsed)
            stamp = time.time()
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)
            elapsed = time.time() - stamp
            print('embedding spent:  %1.4f  s' % elapsed)
            dist = np.sqrt(np.sum(np.square(np.subtract(emb[0,:], emb[1,:]))))
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"distance":str(dist)}).encode())
        else:
            self.send_error(404, 'Not a correct json: %s' % self.path)
        #b+=1
    def handle_http_request(self):
        sendReply = False
        querypath = urlparse(self.path)
        filepath, query = querypath.path, querypath.query
        if filepath.endswith('/'):
            filepath += 'index.html'
        if filepath.endswith(".html"):
            mimetype = 'text/html'
            sendReply = True
        if filepath.endswith(".jpg"):
            mimetype = 'image/jpg'
            sendReply = True
        if filepath.endswith(".gif"):
            mimetype = 'image/gif'
            sendReply = True
        if filepath.endswith(".js"):
            mimetype = 'application/javascript'
            sendReply = True
        if filepath.endswith(".css"):
            mimetype = 'text/css'
            sendReply = True
        if filepath.endswith(".json"):
            mimetype = 'application/json'
            sendReply = True
        if filepath.endswith(".woff"):
            mimetype = 'application/x-font-woff'
            sendReply = True
        if sendReply == True:
            # Open the static file requested and send it
            try:
                with open(path.realpath(curdir + sep + filepath), 'rb') as f:
                    content = f.read()
                    self.send_response(200)
                    self.send_header('Content-type', mimetype)
                    self.end_headers()
                    self.wfile.write(content)
            except IOError:
                self.send_error(404, 'File Not Found: %s' % self.path)

def align_image(image_paths, image_size, margin, pnet, rnet, onet):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    tmp_image_paths=copy.copy(image_paths)
    img_list = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        stamp = time.time()
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
          image_paths.remove(image)
          print("can't detect face, remove ", image)
          continue
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
        elapsed =  time.time()-stamp
        print('align one photo spent:  %1.4f  s' % elapsed)
    images = np.stack(img_list)
    return images

def run():
    
    port = 8081
    print('starting server, port', port)
    # Server settings
    server_address = ('', port)
    httpd = HTTPServer(server_address, testHTTPServer_RequestHandler)
    print('running server...')
    httpd.serve_forever()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',default='../data/models/VGGFace2-0.9965')
    parser.add_argument('-classifier_filename', 
        help='Classifier model file name as a pickle (.pkl) file. ' + 
        'For training this is the output and for classification this is an input.',default='cls.pkl')
    parser.add_argument('-data_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.',default='../data/images/school_0001')
    parser.add_argument('-image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('-margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    return parser.parse_args(argv)
if __name__ == '__main__':
    args=parse_arguments(sys.argv[1:])
    gpu_options = tf.GPUOptions(allow_growth = True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
        print('Preparing mtcnn')
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
        print('done')
        print('Preparing ResNet')
        # Load the model
        facenet.load_model(args.model)
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        print('done')
        print('load svm classifier')
        classifier_filename_exp = os.path.expanduser(args.classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)#读入model 和类名
        print('done')
        a=0
        b=0
        run()
