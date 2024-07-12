# 香橙派连板精度调试记录

板端
```bash
sudo set_device.sh
sudo rknn_server
sudo mkdir /userdata -pv
```

pc 端


```bash
adb kill-server
docker run -it --rm -v ${PWD}:/root/ws -v /dev/bus/usb:/dev/bus/usb --privileged kaylor/rk3588_onnx2rknn:beta bash
```

docker 里面
```bash
这里运行相应的py程序就可以了
```
