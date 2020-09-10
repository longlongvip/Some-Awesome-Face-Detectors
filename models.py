from Haar_Adaboost.model import draw_box_haar as Draw_Haar
from Dlib.model import draw_box_dlib as Draw_Dlib
from CenterFace.model import draw_box_center_face as Draw_CenterFace
from S3FD.model import draw_box_s3fd as Draw_S3FD
from MobileFace.model import draw_box_mobile as Draw_Mobile
from PyramidBox.model import draw_box_pyramid as Draw_Pyramid
# from SSH.model import draw_box_ssh as Draw_SSH
from LFFD.model import draw_box_lffd as Draw_LFFD


class FaceDetector(object):
    def __init__(self, name):
        self.name = name

    def get_results(self, path, image):
        if self.name == 'Haar':
            Draw_Haar(image)
        if self.name == 'Dlib':
            Draw_Dlib(image)
        if self.name == 'CenterFace':
            Draw_CenterFace(image)
        if self.name == 'S3FD':
            Draw_S3FD(image)
        if self.name == 'MobileFace':
            Draw_Mobile(path, image)
        if self.name == 'Pyramid':
            Draw_Pyramid(image)
        # if self.name == 'SSH':
        #     Draw_SSH(path, image)
        if self.name == 'LFFD':
            Draw_LFFD(image)
