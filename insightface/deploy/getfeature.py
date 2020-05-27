import face_model
import argparse
import cv2
import sys
import numpy as np
import math
import pickle
import os
from sklearn.svm import SVC
import facenet
from sklearn.neural_network import MLPClassifier


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')

    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--min_nrof_images_per_class', type=int,
        help='Only include classes with at least this number of images in the dataset', default=15)
    parser.add_argument('--nrof_train_images_per_class', type=int,
        help='Use this number of images from each class for training and the rest for testing', default=10)
    parser.add_argument('--image-size', default='112,112', help='')
    parser.add_argument('--model', default='../models/model-r34-amf/model,0', help='path to load model.')
    parser.add_argument('--ga-model', default='../models/model-r34-amf/model,0', help='path to load model.')
    parser.add_argument('--cpu', default=0, type=int, help='gpu id')
    parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
    parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
    parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
    return parser.parse_args(argv)

def main(args):
    np.random.seed(seed = args.seed)
    dataset_tmp = facenet.get_dataset(args.data_dir)
    train_set, test_set = split_dataset(dataset_tmp, args.min_nrof_images_per_class, args.nrof_train_images_per_class)

    # Check that there are at least one training image per class
    for cls in train_set:
        assert(len(cls.image_paths)>0, 'There must be at least one image for each class in the dataset')            

                 
    train_paths, train_labels = facenet.get_image_paths_and_labels(train_set)
    test_paths, test_labels = facenet.get_image_paths_and_labels(test_set)
    cls_name = [cls.name for cls in train_set]
    with open('cls_name.bin' , 'wb') as fl:
        pickle.dump(cls_name , fl)
            
    train_emb_arr = get_emb_array(train_paths , args)
    test_emb_arr = get_emb_array(test_paths , args)

    with open('train_feature_100.bin' , 'wb') as fl:
        pickle.dump((train_emb_arr , train_labels) , fl)
        print('write trained feature done')
        fl.close()

    with open('test_feature_100.bin' , 'wb') as fl1:
        pickle.dump((test_emb_arr , test_labels) , fl1)
        print('write tested feature done')


def get_emb_array(paths , args):
    insight_model = face_model.FaceModel(args)
    #num_of_images = len(paths)
    emb_array = []
    cnt = 1
    for path in paths:
        if(cnt % 50 == 0):
            print(str(cnt) + '     path:'+path)
        cnt += 1
        img = cv2.imread(path)
        img_tem = insight_model.get_input(img)
        if img_tem is not None:
            img = img_tem
        else:
            img = np.transpose(img , (2 , 0 ,1))

        emb_array.append(insight_model.get_feature(img))

    emb_array = np.array(emb_array)
    return emb_array


def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths)>=min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:min(nrof_train_images_per_class+10 , len(paths))]))
    return train_set, test_set


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
                

    