from tvm.driver import tvmc

# change this if you are not running 2017 macbook pro lol
TARGET = "llvm -mcpu=skylake"

# download model from https://github.com/onnx/models/raw/b9a54e89508f101a1611cd64f4ef56b9cb62c7cf/vision/classification/resnet/model/resnet50-v2-7.onnx
model = tvmc.load("my_model.onnx")

tvmc.tune(model, target=TARGET, trials=1000, early_stopping=100)

package = tvmc.compile(model, target=TARGET)

result = tvmc.run(package, device="cpu")
print(result)
