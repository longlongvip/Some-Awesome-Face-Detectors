import cv2
import torch
import numpy as np
from PyramidBox.pyramid_net import build_pyramid


def get_faces_pyramid(img_pyramid):
    pyramid_model = build_pyramid('test', 2)
    pyramid_model.load_state_dict(torch.load('./PyramidBox/weights/pyramidbox_120000_99.02.pth'))
    pyramid_model.cuda()
    pyramid_model.eval()

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    h, w, _ = img_pyramid.shape
    max_im_shrink = np.sqrt(1050 * 1050 / (h * w))
    img_pyramid_resize = cv2.resize(img_pyramid, None, None, fx=max_im_shrink,
                                    fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)
    # img_s3fd_resize = cv2.resize(img_s3fd, (480, 640))

    x = img_pyramid_resize.swapaxes(1, 2).swapaxes(0, 1)
    x = x.astype('float32')
    x = torch.from_numpy(x).unsqueeze(0)
    if use_cuda:
        x = x.cuda()
    y = pyramid_model(x)
    faces_pyramid = y.data
    scale = torch.tensor([w, h, w, h])
    return faces_pyramid, scale


def draw_box_pyramid(img_pyramid):
    thresh = 0.35
    faces, scale = get_faces_pyramid(img_pyramid)
    print(len(faces))
    for i in range(faces.size(1)):
        j = 0
        while faces[0, i, j, 0] >= thresh:
            pt = (faces[0, i, j, 1:] * scale).cpu().numpy()
            left_up, right_bottom = (pt[0], pt[1]), (pt[2], pt[3])
            j += 1
            cv2.rectangle(img_pyramid, left_up, right_bottom, (0, 0, 255), 2)
