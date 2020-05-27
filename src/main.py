"""Performs face alignment and calculates L2 distance between the embeddings of images."""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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

def main(args):
    print(args.image_files)
    print("args.request=",args.request)
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
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
        embedding_size = embeddings.get_shape()[1]
        print('done')
        print('load svm classifier')
        classifier_filename_exp = os.path.expanduser(args.classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)#读入model 和类名
        print('done')
        #while True:
        if args.request == 'COMPARE':
            stamp = time.time()
            # Run forward pass to calculate embeddings
            images = align_image(args.image_files, args.image_size, args.margin, pnet, rnet, onet, args.gpu_memory_fraction)
            elapsed = time.time() - stamp
            print('align spent:  %1.4f  s' % elapsed)
            stamp = time.time()
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)
            elapsed = time.time() - stamp
            print('embedding spent:  %1.4f  s' % elapsed)
            dist = np.sqrt(np.sum(np.square(np.subtract(emb[0,:], emb[1,:]))))
            print('  %1.4f  ' % dist)


        elif args.request == 'TRAIN':
            dataset = facenet.get_dataset(args.data_dir)#返回ImageClass [(name，path)]
             # Check that there are at least one training image per class
            for cls in dataset:
                assert len(cls.image_paths)>0# 'There must be at least one image for each class in the dataset'
            paths, labels = facenet.get_image_paths_and_labels(dataset)#路径列表，类别列表
            print(labels)
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))
            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i*args.batch_size
                end_index = min((i+1)*args.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, args.image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
            
            
            print('Training classifier')
            model = SVC(kernel='linear', probability=True)
            model.fit(emb_array, labels)
            
            # Create a list of class names
            class_names = [ cls.name.replace('_', ' ') for cls in dataset]
            # Saving classifier model
            with open(classifier_filename_exp, 'wb') as outfile:
                pickle.dump((model, class_names), outfile)
            print('Saved classifier model to file "%s"' % classifier_filename_exp)
        elif args.request == 'CLASSIFY':
            dataset = facenet.get_dataset(args.data_dir)
            for cls in dataset:
                assert len(cls.image_paths)>0# 'There must be at least one image for each class in the dataset'           

                 
            paths, labels = facenet.get_image_paths_and_labels(dataset)#路径列表，类别列表
            print(labels)
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))
            
            
            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i*args.batch_size
                end_index = min((i+1)*args.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, args.image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)

            predictions = model.predict_proba(emb_array)#predictions.shape=(10, 5749)
            best_class_indices = np.argmax(predictions, axis=1)#概率最大的类下标
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]#概率
            
            for i in range(len(best_class_indices)):
                print(labels[i])
                print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
            print(best_class_indices)
            print(labels)
            accuracy = np.mean(np.equal(best_class_indices, labels))
            print('Accuracy: %.3f' % accuracy)
        else:
            print('wrong input')
    
            
def align_image(image_paths, image_size, margin, pnet, rnet, onet, gpu_memory_fraction):

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

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('-classifier_filename', 
        help='Classifier model file name as a pickle (.pkl) file. ' + 
        'For training this is the output and for classification this is an input.',default='cls.pkl')
    parser.add_argument('-image_files', type=str, nargs='*', help='Images to compare')
    parser.add_argument('-data_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.',default='../data/images/school_0001')
    parser.add_argument('-request', type=str,choices=['COMPARE','TRAIN', 'CLASSIFY'],
        help='COMPARE:compare two photos、TRAIN:train classifier、CLASSIFY:classify person')
    parser.add_argument('-image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('-margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('-gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.3)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
