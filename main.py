#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
from src import facenet
#from src.align import detect_face
import math
import pickle
from sklearn.svm import SVC
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from os import path
import urllib
import json
import cgi
import base64
from io import BytesIO
import random
sys.path.append(os.path.join(os.path.dirname(__file__), 'insightface', 'deploy'))
import face_model
import cv2
sys.path.append(os.path.join(os.path.dirname(__file__), 'insightface', 'src','align'))
import detect_face
sys.path.append(os.path.join(os.path.dirname(__file__), 'insightface', 'src','common'))
import face_preprocess

from sklearn.linear_model import SGDClassifier
from multiprocessing import cpu_count

# HTTPRequestHandler class
class school:
    svm_model=0
    class_name=[]
    emb= {}
class testHTTPServer_RequestHandler(BaseHTTPRequestHandler):
    # GET
    
    def do_GET(self):
        print('get')
    def do_POST(self):
        #stamp = time.time()
        if self.headers['content-type']=='application/json':
            global images_placeholder
            global embeddings
            global phase_train_placeholder
            global embedding_size
            global svm_model
            global class_names
            global schools
            
            global model
            length = int(self.headers['content-length'])
            request = json.loads(self.rfile.read(length))
            if request['request'] == 'Compare':
                request['face'][0]=BytesIO(base64.b64decode(request['face'][0]))
                request['face'][1]=BytesIO(base64.b64decode(request['face'][1]))
                print('!!!!!!:',request['face'][0],request['face'][1])
                img1 = misc.imread(request['face'][0], mode='RGB')
                img2 = misc.imread(request['face'][1], mode='RGB')

                img1,box1,_ = cut_face(img1)
                img2,box2,_ = cut_face(img2)
                if len(img1)>1 or len(img2)>1:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'response':'error','description':'There are more than one face in an image'}).encode())

                img1 = img1[0]
                img2 = img2[0]
                f1 = model.get_feature(img1)
                f2 = model.get_feature(img2)
                dist = np.sum(np.square(f1-f2))
#                sim = np.dot(f1, f2.T)
#                print('dist,sim',dist,sim)
                sim = get_sim(dist)

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'response':'ok','similarity':str(sim),'description':''}).encode())


            elif request['request'] == 'AddFace':
                pic=BytesIO(base64.b64decode(request['face']))
                img = misc.imread(pic, mode='RGB')
                images,box,bgr = cut_face(img)
                if len(images) == 1:                    
                    person_path=os.path.join(args.images_dir,request['SchoolId'],request['PersonId'])
                    if not os.path.exists(person_path):
                        os.makedirs(person_path)
                    files = os.listdir(person_path)
                    pic_name = request['PersonId'] + str(len(files)+1) + '.png'
                    pic_path=os.path.join(person_path,pic_name)      
                    cv2.imwrite(pic_path, bgr[0])
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'response':'ok','FileId':os.path.basename(pic_path),'description':''}).encode())
                elif len(images)==0:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'response':'error','FileId':'','description':'no face detected'}).encode())
                    return
                else:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'response':'error','FileId':'','description':'more than one face detected'}).encode())
                    return
                
                
                
            elif request['request'] == 'DelFace':
                response ='ok'
                school_path=os.path.join(args.images_dir,request['SchoolId'])
                if os.path.isdir(school_path):
                    for person in request['PersonId']:
                        person_path=os.path.join(school_path,person[0])
                        if os.path.isdir(person_path):
                            if person[1]=='*':
                                for file_name in os.listdir(person_path):
                                    os.remove(os.path.join(person_path,file_name))
                                os.rmdir(person_path)
                            else:
                                for i in range(1,len(person)):
                                    file_path=os.path.join(person_path,person[i])
                                    if os.path.isfile(file_path):
                                        os.remove(file_path)
                                    else:
                                        response='error'
                        else:
                            response='error'
                else:
                    response='error'
                if response=='ok':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'response':response,'description':''}).encode())
                else:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'response':'error','description':'some or all deletion failed'}).encode())



            elif request['request'] == 'GetFaceList':
                if request['SchoolId'] == '*':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'response':'ok','list':os.listdir(args.images_dir),'description':''}).encode())
                elif request['PersonId'] == '*':
                    path=os.path.join(args.images_dir,request['SchoolId'])
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'response':'ok','list':os.listdir(path),'description':''}).encode())
                elif request['FileId'] == '*':
                    path=os.path.join(args.images_dir,request['SchoolId'],request['PersonId'])
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'response':'ok','list':os.listdir(path),'description':''}).encode())
                else:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'response':'error','list':{},'description':'no such file'}).encode())



            elif request['request'] == 'GetFaceImage':
                path=os.path.join(args.images_dir,request['SchoolId'],request['PersonId'],request['FileId'])
                if os.path.isfile(path):
                    with open(path, 'rb') as f:
                        image=str(base64.b64encode(f.read()))
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'response':'ok','face':image,'description':''}).encode())
                else:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'response':'error','face':'','description':'no such file'}).encode())
                


              
            
            elif request['request'] == 'Train':
                path=os.path.join(args.images_dir,request['SchoolId'])
                if os.path.isdir(path):
                    dataset = facenet.get_dataset(path)
                    if len(dataset)<2:
                        self.send_response(400)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps({'response':'error','description':'number of person less than 2'}).encode())
                        return
                    for cls in dataset:
                        assert len(cls.image_paths)>0 #'There must be at least one image for each class in the dataset'
                    paths, labels = facenet.get_image_paths_and_labels(dataset)
                    print('Number of classes: %d' % len(dataset))
                    print('Number of images: %d' % len(paths))
                    print('Calculating features for images')
                    emb_array = []
                    num_of_images = len(paths)
                    ind = 0
                    for path in paths:
#                        print("path of img " , path)
                        img_tem = misc.imread(path)

                        img,box,_ = cut_face(img_tem)
#                        print("path of img " , len(img))
                        if len(img) < 1 or len(img) > 1:
                            lbl = labels.pop(ind)
                            continue
                        else:
                            ind += 1

                        img = img[0]
                        img_emb = model.get_feature(img)

                        emb_array.append(img_emb)
#                        print('lenth of emb_array' , len(emb_array))
#                    print('lenth of emb_array' , len(emb_array) , '   length of labels' , len(labels))
                    emb_arrar = np.array(emb_array)

                    print('Training classifier')
#                    new_svm_model = SVC(kernel='linear', probability=False)
#                    print(cpu_count())
                    SGDmodel = SGDClassifier(loss="log",max_iter=100, penalty="l2",n_jobs=cpu_count()-1, shuffle=True)
                    SGDmodel.fit(emb_array, labels)
                    new_emb=[]
                    lastidx=-1
                    # Create a list of class names
                    new_class_names = [ cls.name for cls in dataset]
                    for i in range(len(labels)):
                        if labels[i] == lastidx:
                            new_emb[lastidx].append(emb_array[i])
                        else:
                            new_emb.append([emb_array[i]])
                            lastidx=labels[i]
                    #print(emb_array[0:5] , new_emb[0:5])
                    svm_model_path='{}/{}.pkl'.format(args.classifier_dir,request['SchoolId'])
                    # Saving classifier model
                    with open(svm_model_path, 'wb') as outfile:
                        pickle.dump((SGDmodel, new_class_names, new_emb), outfile , protocol = 2)
                    print('Saved classifier to file ',svm_model_path)
                    
                    schools[request['SchoolId']]=school()
                    schools[request['SchoolId']].svm_model=SGDmodel
                    schools[request['SchoolId']].class_names=new_class_names
                    schools[request['SchoolId']].emb=new_emb
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'response':'ok','description':''}).encode())
                else:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'response':'error','description':'no such school'}).encode())


            elif request['request'] == 'Predict':
                if request['SchoolId'] in schools.keys():
                    request['face']=BytesIO(base64.b64decode(request['face']))
                    img1 = misc.imread(request['face'], mode='RGB')
                    images, boxes,_= cut_face(img1)
                    if len(boxes)==0:
                        self.send_response(400)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps({'response':'error','predict':[],'boxes':[],'description':'no face detected'}).encode())
                        return

                    emb = []
                    for image in images:
                        emb.append(model.get_feature(image))
                    emb = np.array(emb)

              
                    predict=[]
                 
                    path=os.path.join(args.images_dir,request['SchoolId'])
                    num_of_people = len(os.listdir(path))
                    
                    predictions = schools[request['SchoolId']].svm_model.predict_proba(emb)  
                    predict_index = np.argmax(predictions,axis=1)
                    best_class_indices = []
                    best_class_probabilities = []
                    count = min(num_of_people,1)
                    for i in range(count):
                        tmp_best_class_indices = np.argmax(predictions, axis=1)
                        tmp_best_class_probabilities = predictions[np.arange(len(tmp_best_class_indices)), tmp_best_class_indices]
                        best_class_indices.append(tmp_best_class_indices)
                        best_class_probabilities.append(tmp_best_class_probabilities)
                        predictions[np.arange(len(tmp_best_class_indices)),tmp_best_class_indices] = 0
                        
                    for i in range(len(best_class_indices[0])):
                        print('%4d' % i)
                        for j in range(len(best_class_indices)):
                            print('j:',j)
                            print('%s: %.3f' % (schools[request['SchoolId']].class_names[best_class_indices[j][i]], best_class_probabilities[j][i]))


                    print('best_class_indices[0]:',best_class_indices[0])
                    similarity=[]
                    for i in range(len(best_class_indices)):
                        tmp = best_class_indices[i]
                        tmp_sim = []
                        for j in range(len(tmp)):
                            dist = 0
                            for k in range(len(schools[request['SchoolId']].emb[tmp[j]])):
                                dist+=np.sum(np.square(np.subtract(emb[j], schools[request['SchoolId']].emb[tmp[j]][k])))
                            dist/=len(schools[request['SchoolId']].emb[tmp[j]])  
                            tmp_sim.append(get_sim(dist))
                        similarity.append(tmp_sim)

                    print('best_class_indices[0]:',best_class_indices[0])
                    for i in range(len(predict_index)):                
                        dist=0
                        for j in range(len(schools[request['SchoolId']].emb[predict_index[i]])):
                            dist+=np.sum(np.square(np.subtract(emb[i], schools[request['SchoolId']].emb[predict_index[i]][j])))
                        dist/=len(schools[request['SchoolId']].emb[predict_index[i]])
                        
                        if dist>args.same_person_threshold:                          
                            predict.append('stranger')
                            
                        else:
                            #tmp = []
                            for k in range(len(best_class_indices)):  
#                                print("?????\n\n\n")
#                                print(len(best_class_indices))                                               
                                predict.append([schools[request['SchoolId']].class_names[best_class_indices[k][i]],similarity[k][i]])
                            #predict.append(tmp)
                            
                    print('predict' , predict)
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'response':'ok','predict':predict,'boxes':boxes,'description':''}).encode())
                else:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'response':'error','predict':[],'boxes':[],'description':'no such school'}).encode())
        else:
            self.send_error(404, 'Not a correct json: %s' % self.path)
        #print('server time',time.time()-stamp)

def align_image_no_prewhitened(image_objects, image_size, margin, pnet, rnet, onet):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    img_list = []
    for image in image_objects:
        img = misc.imread(image, mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
            image_objects.remove(image)
            print('can\'t detect face, remove ', image)
            continue
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        #prewhitened = facenet.prewhiten(aligned)
        #img_list.append(prewhitened)
        img_list.append(aligned)
    if len(img_list)==0:
        return img_list
    images = np.stack(img_list)
    return images


def align_image(image_objects, image_size, margin, pnet, rnet, onet):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    img_list = []

    for image in image_objects:
        img = misc.imread(image, mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
            print('can\'t detect face')
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
    if len(img_list)==0:
        return img_list
    images = np.stack(img_list)
    return images 

def align_image2(image, image_size, margin, pnet, rnet, onet):
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    img_list = []
    boxes=[]

    img = misc.imread(image, mode='RGB')
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    #print('number of box',len(bounding_boxes))
    if len(bounding_boxes) < 1:
        print('can\'t detect face')
        return img_list,boxes
    for det in bounding_boxes:
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        boxes.append(bb.tolist())
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images,boxes

def load_data(image_paths,image_size):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in range(nrof_samples):
        images[i,:,:,:] = facenet.prewhiten(misc.imread(image_paths[i]))
    return images
def run():
    print('starting server, port', args.port)
    # Server settings
    server_address = ('0.0.0.0', args.port)
    httpd = HTTPServer(server_address, testHTTPServer_RequestHandler)
    print('running server...')
    httpd.serve_forever()
    
def get_sim(dist):
    if dist <= 0.8:
        return 1
    elif dist >= 2.3:
        return 0
    elif dist >0.8 and dist <1.0:
        return -0.5*dist + 1.4
    else:
        return -0.667*dist + 1.5336

def get_bound(x1,y1,x2,y2):
    k=(y2-y1)/(x2-x1)
    b=y1-k*x1
    return k,b
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-embedding_model_dir', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',default='data/models/face_model6')
    parser.add_argument('-classifier_dir', 
        help='Classifier model file name as a pickle (.pkl) file. ' + 
        'For training this is the output and for classification this is an input.',default='data/models/svm')
    parser.add_argument('-backup_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.',default='data/backup')
    parser.add_argument('-images_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.',default='data/images')
#    parser.add_argument('-image_size', type=int,
#        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('-image_size', type=str,
        help='Image size (height, width) in pixels.', default='112,112')
    parser.add_argument('-margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=32)
    parser.add_argument('-batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('-port', type=int,
        help='server port', default=8849)
    parser.add_argument('-same_person_threshold', type=float,
        help='the bound of determine as same person for compare', default=1)
    
    
#    parser.add_argument('--image-size', default='112,112', help='')
    parser.add_argument('--model', default='insightface/models/model-r34-amf/model,0', help='path to load model.')
    parser.add_argument('--ga-model', default='insightface/models/model-r34-amf/model,0', help='path to load model.')
    parser.add_argument('--cpu', default=0, type=int, help='cpu id')
    parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
    parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
    parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
    
    return parser.parse_args(argv)


################################   new     #######################################
def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def IOU(Reframe,GTframe):
  x1 = Reframe[0];
  y1 = Reframe[1];
  width1 = Reframe[2]-Reframe[0];
  height1 = Reframe[3]-Reframe[1];

  x2 = GTframe[0]
  y2 = GTframe[1]
  width2 = GTframe[2]-GTframe[0]
  height2 = GTframe[3]-GTframe[1]

  endx = max(x1+width1,x2+width2)
  startx = min(x1,x2)
  width = width1+width2-(endx-startx)

  endy = max(y1+height1,y2+height2)
  starty = min(y1,y2)
  height = height1+height2-(endy-starty)

  if width <=0 or height <= 0:
    ratio = 0
  else:
    Area = width*height
    Area1 = width1*height1
    Area2 = width2*height2
    ratio = Area*1./(Area1+Area2-Area)
  return ratio


def cut_face(img):  
    minsize = 20
    threshold = [0.6,0.7,0.9]
    factor = 0.709
    
    nrof = np.zeros( (5,), dtype=np.int32)
    
#    try:
#        img = misc.imread(image_path)
#    except (IOError, ValueError, IndexError) as e:
#        errorMessage = '{}: {}'.format(image_path, e)
#        print(errorMessage)
#    else:
#    if img.ndim<2:
#        print('Unable to align "%s", img dim error' % image_path)
#        return
    if img.ndim == 2:
        img = to_rgb(img)
    img = img[:,:,0:3]
#    target_dir = output_dir
#    
#    if not os.path.exists(target_dir):
#      os.makedirs(target_dir)
    _minsize = minsize
    bounding_boxes, points = detect_face.detect_face(img, _minsize, pnet, rnet, onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]
    
    all_bbox = []
    if nrof_faces>0:
      
      all_landmark = []
      for i in range(nrof_faces):
          all_bbox.append(bounding_boxes[i, 0:4].tolist())
          all_landmark.append(points[:, i].reshape( (2,5) ).T)

      nrof[0]+=1
    else:
      nrof[1]+=1
     
    images = []
    bgr = []
    for i in range(nrof_faces):    
        warped = face_preprocess.preprocess(img, bbox=all_bbox[i], landmark = all_landmark[i], image_size='112,112')
        bgr.append(warped[...,::-1])
#        name = output_dir + '/' + str(i)+'.jpg'
#        cv2.imwrite(name, bgr)
        
        nimg = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(nimg, (2,0,1))
        images.append(aligned)

    return images,all_bbox,bgr

################################   new     #######################################


if __name__ == '__main__':
    random.seed(time.time())
    args=parse_arguments(sys.argv[1:])
    k1,b1=get_bound(0,1,args.same_person_threshold,0.8)
    k2,b2=get_bound(args.same_person_threshold,0.8,4,0)
    
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            print('loading mtcnn...')
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
            print('done.')
   
    print('loading model...')
    model = face_model.FaceModel(args)    
    print('done.')

    print('loading classifier...')
    
    schools={}
    
    svm_list=os.listdir(args.classifier_dir)
    for svm_file in svm_list:
        svm_file2 = svm_file.split('.')
        if len(svm_file2)!=2 or svm_file2[1]!='pkl':#命名格式含多个点或后缀不等于pkl
            print('skip loading file ',svm_file,' as svm: name format error')
            continue
        with open(os.path.join(args.classifier_dir,svm_file), 'rb') as infile:
            schools[svm_file2[0]]=school()
            school_data = pickle.load(infile)
        schools[svm_file2[0]].svm_model = school_data[0]
        schools[svm_file2[0]].class_names = school_data[1]
        schools[svm_file2[0]].emb = school_data[2]#读入model 和类名
    print('done.')
    
   
    run()
    
