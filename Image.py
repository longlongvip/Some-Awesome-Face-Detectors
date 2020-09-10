from torchvision.transforms import transforms
from PIL import Image
import cv2

path = 'Face/Images/2.jpg'

img_pil = Image.open(path)  # PIL RGB <--> WH
img_cv2 = cv2.imread(path)  # numpy.ndarray BGR <--> HWC
img_cv2_hwc = img_cv2  # numpy.ndarray BGR <--> HWC

"""PIL or numpy.ndarray to Tensor"""
img_pil_tensor = transforms.ToTensor()(img_pil)  # Tensor RGB <--> CHW
img_cv2_tensor = transforms.ToTensor()(img_cv2)  # Tensor BGR <--> CHW

"""Tensor or numpy.ndarray to PIL"""
img_tensor_pil = transforms.ToPILImage()(img_pil_tensor)  # PIL RGB <--> WH
img_cv2_pil = transforms.ToPILImage()(img_cv2)  # PIL BGR <--> WH

"""PIL or Tensor to numpy.ndarray"""
img_pilTensor_numpy = img_pil_tensor.numpy()  # numpy.ndarray RGB <--> CHW
img_cv2Tensor_numpy = img_cv2_tensor.numpy()  # numpy.ndarray BGR <--> CHW

"""PIL <--> CHW"""
img_pil_tensor_chw = transforms.ToTensor()(img_pil)  # Tensor RGB <--> CHW
img_pil_numpy_chw = img_pil_tensor_chw.numpy()  # numpy.ndarray RGB <--> CHW

"""HWC <--> CHW"""
"""OpenCV <--> numpy.ndarray"""
img_cv2_chw = img_cv2_hwc.transpose((2, 0, 1))  # numpy.ndarray BGR <--> CHW

"""CHW <--> HWC"""
"""OpenCV <--> numpy.ndarray"""
img_cv2_hwc = img_cv2_chw.transpose((1, 2, 0))  # numpy.ndarray BGR <--> HWC
"""Tensor"""
# Tensor(CHW) --> numpy.ndarray(CHW) --> transpose((1, 2, 0)) --> HWC
