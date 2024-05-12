def detect_forward(self, x):
    shape = x[0].shape
    y = []
    for i in range(self.nl):
        y.append(self.cv2[i](x[i]))
        y.append(self.cv3[i](x[i]))
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
