import math
import torch
from ultralytics.utils.tal import make_anchors
def detect_forward(self, x):
    shape = x[0].shape
    if self.dynamic or self.shape != shape:
            print(self.stride)
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x, self.stride, 0.5))
            self.shape = shape
    y = []
    for i in range(self.nl):
        y.append(self.cv2[i](x[i]))
        y.append(torch.sigmoid(self.cv3[i](x[i])))
    print(f'self.anchors shape: {self.anchors.shape}')
    return y


def segment_forward(self, x):
    p = self.proto(x[0])  # mask protos
    bs = p.shape[0]  # batch size
    mc = [self.cv4[i](x[i]) for i in range(self.nl)]
    x = self.detect(self, x)
    bo = len(x) // 3
    relocated = []
    for i in range(len(mc)):
        relocated.extend(x[i * bo : (i + 1) * bo])
        relocated.extend([mc[i]])
    relocated.extend([p])
    return relocated

def pose_forward(self, x):
    bs = x[0].shape[0]
    kpt = [self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)]
    x = self.detect(self, x)
    pred_kpts = self.kpt_decode(bs, kpt);
    bo = len(x) // 3
    relocated = []
    for i in range(len(pred_kpts)):
         relocated.extend(x[i*bo : (i + 1) * bo])
         relocated.extend([pred_kpts[i]])
    return relocated

def pose_kpt_decode(self, bs, kpts):
    ndim = self.kpt_shape[1]
    # print(self.kpt_shape)
    anch_len = []
    anch_len.append(0)
    pred_kpts = []
    for i,kpt in enumerate(kpts):
         anch_len.append(kpt.shape[2])
         print(f'kpt shape: {kpt.shape}')
         y = kpt.view(bs, *self.kpt_shape, -1)
         print(f'y shape: {y.shape}, self.anchors shape: {self.anchors.shape}, self.strides shape: {self.strides.shape}')
        #  print(f'--- shape : {y[:, :, :2].shape}')
         start_idx = anch_len[i];
         end_idx = anch_len[i + 1];
    
         a = (y[:, :, :2] * 2.0 + (self.anchors[:, start_idx:end_idx] - 0.5)) * self.strides[:, start_idx:end_idx]
         if ndim == 3:
            a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
         pred_kpts.extend([a.view(bs, self.nk, -1)])



def obb_forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        bs = x[0].shape[0]  # batch size
        # angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # OBB theta logits
        angle = [self.cv4[i](x[i]) for i in range(self.nl)]
        # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        # angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
        # angle = angle.sigmoid() * math.pi / 2  # [0, pi/2]
        x = self.detect(self, x)
        bo = len(x) // 3
        relocated = []
        for i in range(len(angle)):
            relocated.extend(x[i * bo: (i + 1) * bo])
            tmp = (angle[i].sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
            relocated.extend([tmp])
        return relocated


