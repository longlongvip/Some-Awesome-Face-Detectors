import cv2


def show(title, img, t=1000, esc=True):
    """
    该方法实现的功能为，当esc属性为False的时候，窗口显示tms自动关闭；
    而当esc为True的时候，我们会检测窗口是否存在和是否按下ESC键，
    若手动关闭了窗口或按下ESC键关闭窗口，程序将继续运行。
    """
    cv2.namedWindow(title, 0)
    cv2.imshow(title, img)
    if esc:
        while cv2.waitKey(100) != 27:
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) <= 0:
                break
    else:
        cv2.waitKey(t)
    cv2.destroyWindow(title)
