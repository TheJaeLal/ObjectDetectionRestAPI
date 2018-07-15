import os

#Parameters
threshold_score = 0.20

#Stream Url
#stream_url = "http://192.168.43.1:8080/shot.jpg"

#Relative Paths...
std_model_infer_path = os.path.join("model","standard","frozen_inference_graph.pb")
mask_model_infer_path = os.path.join("model","mask_rcnn","frozen_inference_graph.pb")
class_map_file = os.path.join('model','class_names.csv')

#For static image object detection..
test_imgs_dir = 'Test_Images'
result_imgs_dir = 'Results'
