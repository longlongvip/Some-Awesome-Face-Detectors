import cv2
import torch
import numpy as np
from S3FD.s3fd_net import build_s3fd


def get_faces_s3fd(img_s3fd):
    s3fd_model = build_s3fd('test', 2)
    s3fd_model.load_state_dict(torch.load('./S3FD/weights/s3fd_face.pth'))
    s3fd_model.cuda()
    s3fd_model.eval()

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    h, w, _ = img_s3fd.shape
    max_im_shrink = np.sqrt(1700 * 1200 / (h * w))
    img_s3fd_resize = cv2.resize(img_s3fd, None, None, fx=max_im_shrink,
                                 fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)
    # img_s3fd_resize = cv2.resize(img_s3fd, (480, 640))

    x = img_s3fd_resize.swapaxes(1, 2).swapaxes(0, 1)
    x = x.astype('float32')
    x = torch.from_numpy(x).unsqueeze(0)
    if use_cuda:
        x = x.cuda()
    y = s3fd_model(x)
    faces_s3fd = y.data
    scale = torch.tensor([w, h, w, h])
    return faces_s3fd, scale


def draw_box_s3fd(img_s3fd):
    thresh = 0.35
    faces, scale = get_faces_s3fd(img_s3fd)
    print(len(faces))
    for i in range(faces.size(1)):
        j = 0
        while faces[0, i, j, 0] >= thresh:
            pt = (faces[0, i, j, 1:] * scale).cpu().numpy()
            left_up, right_bottom = (pt[0], pt[1]), (pt[2], pt[3])
            j += 1
            cv2.rectangle(img_s3fd, left_up, right_bottom, (0, 0, 255), 2)
