"""
BSD 3-Clause License

Copyright (c) 2017, Adrian Bulat
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
try:
    from .net_s3fd import s3fd
except ImportError:
    from net_s3fd import s3fd

class SFDDetector(object):

    __WHITENING = np.array([104, 117, 123])

    def __init__(self, device, path_to_detector=None):
        self.device = device

        if path_to_detector is None:
            import rospkg
            path_to_detector = rospkg.RosPack().get_path('rt_gene') + '/model_nets/SFD/s3fd_facedetector.pth'

        if 'cuda' in device:
            torch.backends.cudnn.benchmark = True

        self.face_detector = s3fd()
        self.face_detector.load_state_dict(torch.load(path_to_detector))
        self.face_detector.to(device)
        self.face_detector.eval()

    def detect_from_image(self, tensor_or_path):
        image = self.tensor_or_path_to_ndarray(tensor_or_path)

        bboxlist = self.detect(self.face_detector, image, device=self.device)
        keep = self.nms(bboxlist, 0.3)
        bboxlist = bboxlist[keep, :]
        bboxlist = [x for x in bboxlist if x[-1] > 0.5]

        return bboxlist

    @staticmethod
    def tensor_or_path_to_ndarray(tensor_or_path, rgb=True):
        """Convert path (represented as a string) or torch.tensor to a numpy.ndarray

        Arguments:
            tensor_or_path {numpy.ndarray, torch.tensor or string} -- path to the image, or the image itself
        """
        if isinstance(tensor_or_path, str):
            from skimage import io
            return cv2.imread(tensor_or_path) if not rgb else io.imread(tensor_or_path)
        elif torch.is_tensor(tensor_or_path):
            # Call cpu in case its coming from cuda
            return tensor_or_path.cpu().numpy()[..., ::-1].copy() if not rgb else tensor_or_path.cpu().numpy()
        elif isinstance(tensor_or_path, np.ndarray):
            return tensor_or_path[..., ::-1].copy() if not rgb else tensor_or_path
        else:
            raise TypeError

    @staticmethod
    def nms(dets, thresh):
        if 0 == len(dets):
            return []
        x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
            xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])

            w, h = np.maximum(0.0, xx2 - xx1 + 1), np.maximum(0.0, yy2 - yy1 + 1)
            ovr = w * h / (areas[i] + areas[order[1:]] - w * h)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

    @staticmethod
    def decode(loc, priors, variances):
        """Decode locations from predictions using priors to undo
        the encoding we did for offset regression at train time.
        Args:
            loc (tensor): location predictions for loc layers,
                Shape: [num_priors,4]
            priors (tensor): Prior boxes in center-offset form.
                Shape: [num_priors,4].
            variances: (list[float]) Variances of priorboxes
        Return:
            decoded bounding box predictions
        """

        boxes = torch.cat((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def detect(self, net, img, device):
        img = img - self.__WHITENING
        img = img.transpose(2, 0, 1)
        img = img.reshape((1,) + img.shape)

        img = torch.from_numpy(img).float().to(device)
        with torch.no_grad():
            olist = net(img)

        bboxlist = []
        for i in range(len(olist) // 2):
            olist[i * 2] = F.softmax(olist[i * 2], dim=1)
        olist = [oelem.data.cpu() for oelem in olist]
        for i in range(len(olist) // 2):
            ocls, oreg = olist[i * 2], olist[i * 2 + 1]
            stride = 2 ** (i + 2)  # 4,8,16,32,64,128

            poss = zip(*np.where(ocls[:, 1, :, :] > 0.05))
            for Iindex, hindex, windex in poss:
                axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
                score = ocls[0, 1, hindex, windex]
                loc = oreg[0, :, hindex, windex].contiguous().view(1, 4)
                priors = torch.Tensor([[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]])
                variances = [0.1, 0.2]
                box = self.decode(loc, priors, variances)
                x1, y1, x2, y2 = box[0] * 1.0
                bboxlist.append([x1, y1, x2, y2, score])
        bboxlist = np.array(bboxlist)
        if 0 == len(bboxlist):
            bboxlist = np.zeros((1, 5))

        return bboxlist

class LandmarkMethodBase(object):
    def __init__(self, device_id_facedetection, checkpoint_path_face=None, checkpoint_path_landmark=None, model_points_file=None):
        download_external_landmark_models()
        self.model_size_rescale = 16.0
        self.head_pitch = 0.0
        self.interpupillary_distance = 0.058
        self.eye_image_size = (60, 36)

        tqdm.write("Using device {} for face detection.".format(device_id_facedetection))

        self.device = device_id_facedetection
        self.face_net = SFDDetector(device=device_id_facedetection, path_to_detector=checkpoint_path_face)
        self.facial_landmark_nn = self.load_face_landmark_model(checkpoint_path_landmark)

        self.model_points = self.get_full_model_points(model_points_file)

    def load_face_landmark_model(self, checkpoint_fp=None):
        import rt_gene.ThreeDDFA.mobilenet_v1 as mobilenet_v1
        if checkpoint_fp is None:
            import rospkg
            checkpoint_fp = rospkg.RosPack().get_path('rt_gene') + '/model_nets/phase1_wpdc_vdc.pth.tar'
        arch = 'mobilenet_1'

        checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
        model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)

        model_dict = model.state_dict()
        # because the model is trained by multiple gpus, prefix module should be removed
        for k in checkpoint.keys():
            model_dict[k.replace('module.', '')] = checkpoint[k]
        model.load_state_dict(model_dict)
        cudnn.benchmark = True
        model = model.to(self.device)
        model.eval()
        return model

    def get_full_model_points(self, model_points_file=None):
        """Get all 68 3D model points from file"""
        raw_value = []
        if model_points_file is None:
            import rospkg
            model_points_file = rospkg.RosPack().get_path('rt_gene') + '/model_nets/face_model_68.txt'

        with open(model_points_file) as f:
            for line in f:
                raw_value.append(line)
        model_points = np.array(raw_value, dtype=np.float32)
        model_points = np.reshape(model_points, (3, -1)).T

        # index the expansion of the model based.
        model_points = model_points * (self.interpupillary_distance * self.model_size_rescale)

        return model_points

    def get_init_value(self, image):
        faceboxes = self.get_face_bb(image)
        bbox_l = []
        for facebox in faceboxes:
            bbox_l.append(facebox[0])

        return len(faceboxes), bbox_l

    def get_face_bb(self, image):
        faceboxes = []
        fraction = 4.0
        image = cv2.resize(image, (0, 0), fx=1.0 / fraction, fy=1.0 / fraction)
        detections = self.face_net.detect_from_image(image)

        for result in detections:
            # scale back up to image size
            box = result[:4]
            confidence = result[4]

            if gaze_tools.box_in_image(box, image) and confidence > 0.6:
                box = [x * fraction for x in box]  # scale back up
                diff_height_width = (box[3] - box[1]) - (box[2] - box[0])
                offset_y = int(abs(diff_height_width / 2))
                box_moved = gaze_tools.move_box(box, [0, offset_y])

                # Make box square.
                facebox = gaze_tools.get_square_box(box_moved)
                faceboxes.append(facebox)

        return faceboxes


def detect(self, net, img, device):
    img = img - self.__WHITENING
    img = img.transpose(2, 0, 1)
    img = img.reshape((1,) + img.shape)

    img = torch.from_numpy(img).float().to(device)
    with torch.no_grad():
        olist = net(img)

    bboxlist = []
    for i in range(len(olist) // 2):
        olist[i * 2] = F.softmax(olist[i * 2], dim=1)
    olist = [oelem.data.cpu() for oelem in olist]
    for i in range(len(olist) // 2):
        ocls, oreg = olist[i * 2], olist[i * 2 + 1]
        stride = 2 ** (i + 2)  # 4,8,16,32,64,128

        poss = zip(*np.where(ocls[:, 1, :, :] > 0.05))
        for Iindex, hindex, windex in poss:
            axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
            score = ocls[0, 1, hindex, windex]
            loc = oreg[0, :, hindex, windex].contiguous().view(1, 4)
            priors = torch.Tensor([[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]])
            variances = [0.1, 0.2]
            box = self.decode(loc, priors, variances)
            x1, y1, x2, y2 = box[0] * 1.0
            bboxlist.append([x1, y1, x2, y2, score])
    bboxlist = np.array(bboxlist)
    if 0 == len(bboxlist):
        bboxlist = np.zeros((1, 5))

    return bboxlist

def move_box(box, offset):
    """Move the box to direction specified by vector offset"""
    left_x = box[0] + offset[0]
    top_y = box[1] + offset[1]
    right_x = box[2] + offset[0]
    bottom_y = box[3] + offset[1]

    return [left_x, top_y, right_x, bottom_y]

def box_in_image(box, image):
    """Check if the box is in image"""
    rows = image.shape[0]
    cols = image.shape[1]

    return box[0] >= 0 and box[1] >= 0 and box[2] <= cols and box[3] <= rows

def get_square_box(box):
    """Get a square box out of the given box, by expanding it."""
    left_x = box[0]
    top_y = box[1]
    right_x = box[2]
    bottom_y = box[3]

    box_width = right_x - left_x
    box_height = bottom_y - top_y

    # Check if box is already a square. If not, make it a square.
    diff = box_height - box_width
    delta = int(abs(diff) / 2)

    if diff == 0:  # Already a square.
        return box
    elif diff > 0:  # Height > width, a slim box.
        left_x -= delta
        right_x += delta
        if diff % 2 == 1:
            right_x += 1
    else:  # Width > height, a short box.
        top_y -= delta
        bottom_y += delta
        if diff % 2 == 1:
            bottom_y += 1

    return [left_x, top_y, right_x, bottom_y]


