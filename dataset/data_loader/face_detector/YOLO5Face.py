from copy import deepcopy
import os
import cv2
import torch
import torch.nn as nn
from dataset.data_loader.face_detector.model.yolo import Model
from dataset.data_loader.face_detector.utils.data_ops import letterbox, scale_coords_landmarks, show_results, check_img_size, non_max_suppression_face, scale_coords


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    # ch_in, ch_out, kernel, stride, padding, groups
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p),
                              groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = self.act = nn.LeakyReLU(0.1, inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class YOLO5Face(object):
    def __init__(self, backend="Y5F", device=None) -> None:
        self.conf_thres = 0.6
        self.iou_thres = 0.5
        self.imgsz = (640, 640)
        self.img_size = 640

        if device != None:
            if device == "cpu":
                self.device = torch.device("cpu")
            elif torch.cuda.is_available():
                dev_list = [int(d) for d in device.replace("cuda:", "").split(",")]
                self.device = torch.device(dev_list[0])     #currently toolbox only supports 1 GPU
            else:
                self.device = torch.device("cpu")
        else:
            if torch.cuda.is_available():
                self.device = torch.device(0)     #currently toolbox only supports 1 GPU
            else:
                self.device = torch.device("cpu")

        package_dir = os.path.dirname(os.path.abspath(__file__))
        model_config_path = os.path.join(package_dir, "model", "yolo5face.yaml")

        if backend == "Y5F_IR":
            ckpt = os.path.join(package_dir, "ckpts", "Y5sF_iBVP_IR.pth")
            self.model = Model(cfg=model_config_path, ch=1, nc=None).to(self.device)
        else:
            self.model = Model(cfg=model_config_path, ch=3, nc=None).to(self.device)
            ckpt = os.path.join(package_dir, "ckpts", "Y5sF_WFRGB.pth")

        checkpoint = torch.load(ckpt, map_location=self.device)
        self.model.load_state_dict(checkpoint)

        # Compatibility updates
        for m in self.model.modules():
            if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True  # pytorch 1.7.0 compatibility
            elif type(m) is Conv:
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

        self.model.float().fuse().eval()


    def detect_face(self, frame):

        img0 = deepcopy(frame)
        h0, w0 = img0.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        imgsz = check_img_size(self.img_size, s=self.model.stride.max())  # check img_size
        img = letterbox(img0, new_shape=imgsz)[0]

        # Convert from w,h,c to c,w,h
        img = img.transpose(2, 0, 1).copy()

        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            # Inference
            pred = self.model(img)[0]
            
            # Apply NMS
            pred = non_max_suppression_face(pred, self.conf_thres, self.iou_thres)

            if len(pred[0]) > 0:
                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    if i==0 and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class

                        det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], frame.shape).round()

                        # only process first face.
                        # for j in range(det.size()[0]):
                        xyxy = det[0, :4].view(-1).tolist()

                        x1 = int(xyxy[0])
                        y1 = int(xyxy[1])
                        x2 = int(xyxy[2])
                        y2 = int(xyxy[3])
                        res = [x1, y1, x2, y2]
            else:
                res = None
            
            return res