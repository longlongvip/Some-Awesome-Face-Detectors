import cv2
import dlib


def get_faces_dlib(img_dlib):
    dlib_model = dlib.get_frontal_face_detector()
    faces_dlib = dlib_model(img_dlib)
    return faces_dlib


def draw_box_dlib(img_dlib):
    faces = get_faces_dlib(img_dlib)
    if not faces or len(faces) < 1:
        print("No faces detection")
    else:
        for k, box in enumerate(faces):
            face = [(box.left(), box.top()), (box.right(), box.bottom())]
            cv2.rectangle(img_dlib, face[0], face[1], (0, 0, 255), 2)
