import tensorflow as tf

import numpy as np
import os

import utils
import config


detection_graph = tf.Graph()


with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(config.std_model_infer_path,mode='rb') as graph_file:
        serialized_graph = graph_file.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def)


class_map = utils.get_class_map(config.class_map_file)


def run(test_img):

    # #Load the input image
    # img_path = os.path.join(config.test_imgs_dir,img_name)
    # test_img = utils.load_image(img_path)
    
    with tf.Session(graph = detection_graph) as sess:

    #Get access to the relevant input and output tensors
        input_ = sess.graph.get_tensor_by_name("import/image_tensor:0")
        boxes = sess.graph.get_tensor_by_name("import/detection_boxes:0")
        scores = sess.graph.get_tensor_by_name("import/detection_scores:0")
        classes = sess.graph.get_tensor_by_name("import/detection_classes:0")
        num_detections = sess.graph.get_tensor_by_name("import/num_detections:0")

        (boxes, scores, classes, num_detections) = sess.run( 
                        [boxes, scores, classes, num_detections],
                        feed_dict={input_: test_img})

        boxes = np.squeeze(boxes,axis=0)
        scores = np.squeeze(scores,axis=0)
        classes = np.squeeze(classes,axis=0)
        test_img = np.squeeze(test_img,axis=0)

        detections = utils.get_detections(scores,config.threshold_score)

        utils.draw_bounding_box(test_img,detections,boxes,classes,class_map)
    
    return test_img
    # out_path = os.path.join(config.result_imgs_dir,img_name)
    # utils.save_image(out_path,test_img)