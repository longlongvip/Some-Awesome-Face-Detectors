import os
from base_module import *
from models import FaceDetector as FD

detector_name = 'LFFD'
detector_mode = "Image"

if detector_mode == "Image":
    img_path = "Face/Images/3.jpg"
    img = cv2.imread(img_path)
    detector = FD(detector_name)
    detector.get_results(path=img_path, image=img)
    show("Face Box", img)

if detector_mode == "Image_List":
    img_list_path = "Face/Images/"
    img_list = [os.path.join(img_list_path, x) for x in os.listdir(img_list_path) if x.endswith('jpg')]
    for path in img_list:
        img = cv2.imread(path)
        detector = FD(detector_name)
        detector.get_results(img)
        show("Face Box", img)
