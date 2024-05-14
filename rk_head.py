import math
import torch
def detect_forward(self, x):
    shape = x[0].shape
    y = []
    for i in range(self.nl):
        y.append(self.cv2[i](x[i]))
        y.append(torch.sigmoid(self.cv3[i](x[i])))
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


