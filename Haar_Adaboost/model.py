import cv2


def get_faces_haar(img_haar):
    haar_model = cv2.CascadeClassifier('./Haar_Adaboost/haar_cascade_face.xml')
    # 一定要使用整个工程的路径
    # lbp_model = cv2.CascadeClassifier('lbp_cascade_face.xml')
    faces_haar = haar_model.detectMultiScale(img_haar)
    return faces_haar


def draw_box_haar(img_haar):
    faces = get_faces_haar(img_haar)
    if not faces or len(faces) < 1:
        print("No faces detection")
    else:
        for (x, y, w, h) in faces:
            cv2.rectangle(img_haar, (x, y), (x + w, y + h), (0, 0, 255), 2)
