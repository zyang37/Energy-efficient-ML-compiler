import onnx
from tvm.contrib.download import download_testdata
from PIL import Image
import numpy as np
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor

# pick your flavor of resnet<XX>-v2-7.onnx
# https://github.com/onnx/models/tree/main/vision/classification/resnet/model
RESNET = 34
tuning_records = f"resnet-{RESNET}-v2-autotuning_with_energy.json",

model_url = (
    f"https://github.com/onnx/models/raw/main/"
    f"vision/classification/resnet/model/"
    f"resnet{RESNET}-v2-7.onnx"
)

model_path = download_testdata(model_url, f"resnet{RESNET}-v2-7.onnx", module="onnx")
onnx_model = onnx.load(model_path)

# Seed numpy's RNG to get consistent results
np.random.seed(0)

################################################################################
# Downloading, Preprocessing, and Loading the Test Image


img_url = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
img_path = download_testdata(img_url, "imagenet_cat.png", module="data")

# Resize it to 224x224
resized_image = Image.open(img_path).resize((224, 224))
img_data = np.asarray(resized_image).astype("float32")

# Our input image is in HWC layout while ONNX expects CHW input, so convert the array
img_data = np.transpose(img_data, (2, 0, 1))

# Normalize according to the ImageNet input specification
imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
imagenet_stddev = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
norm_img_data = (img_data / 255 - imagenet_mean) / imagenet_stddev

# Add the batch dimension, as we are expecting 4-dimensional input: NCHW.
img_data = np.expand_dims(norm_img_data, axis=0)

###############################################################################
# Compile the Model With Relay

target = "llvm -mcpu=skylake"

######################################################################
# .. admonition:: Defining the Correct Target

input_name = "data"
shape_dict = {input_name: img_data.shape}

mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))


######################################################################
# Count the total model of floating point operations within this model
from tvm import autotvm

total_flop_count = 0
tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)
for i, task in enumerate(tasks):
    total_flop_count += task.flop


######################################################################
# Execute on the TVM Runtime

dtype = "float32"
module.set_input(input_name, img_data)
module.run()
output_shape = (1, 1000)
tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()

################################################################################
# Collect Basic Performance Data

import timeit
from pyJoules.device import DeviceFactory
from pyJoules.device.rapl_device import RaplPackageDomain, RaplDramDomain, RaplCoreDomain, RaplUncoreDomain
from pyJoules.energy_meter import EnergyMeter

# manually construct energy meter. Refer to docs here: https://pyjoules.readthedocs.io/en/latest/usages/manual_usage.html
domains = [RaplPackageDomain(0), RaplUncoreDomain(0), RaplDramDomain(0)]
devices = DeviceFactory.create_devices(domains)
meter = EnergyMeter(devices)

timing_number = 100  # number of times to run the model in a single timing loop
timing_repeat = 10  # number of times to repeat the timing loop (length of the results array)

meter.start()
raw_results = timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number)
meter.stop()

# get the energy consumption
trace = meter.get_trace()
sample = trace[0] # only one sample covering the entire period, as no "hotspots" were specified
# I take the package domain and subtract integrated graphics, the add DRAM as well. You can refer to the following diagram for explanation:
# https://pyjoules.readthedocs.io/en/latest/devices/intel_cpu.html#domains
total_energy_uJ = (sample.energy['package_0'] - sample.energy['uncore_0'] + sample.energy['dram_0'])
total_energy_J = total_energy_uJ / 1e6
average_energy_per_inference_J = total_energy_J / timing_number

unoptimized_times_s = (
    np.array(raw_results)
    / timing_number
)
unoptimized_flops = total_flop_count / unoptimized_times_s
unoptimized_watts = average_energy_per_inference_J / unoptimized_times_s
unoptimized_flops_per_watt = unoptimized_flops / unoptimized_watts
unoptimized_gflops_per_watt = unoptimized_flops_per_watt / 1e9  # convert to gigaflops

unoptimized_stats_seconds = {
    "mean": np.mean(unoptimized_times_s),
    "median": np.median(unoptimized_times_s),
    "std": np.std(unoptimized_times_s),
}
unoptimized_stats_gflops_per_watt = {
    "mean": np.mean(unoptimized_gflops_per_watt),
    "median": np.median(unoptimized_gflops_per_watt),
    "std": np.std(unoptimized_gflops_per_watt),
}

# ointervalintervaltime in milliseconds!
print("unoptimized time: %s" % (unoptimized_stats_seconds))
print("unoptimized gflops/watt: %s" % (unoptimized_stats_gflops_per_watt))


################################################################################
# Compiling an Optimized Model with Tuning Data
# ----------------------------------------------

with autotvm.apply_history_best(tuning_records):
    with tvm.transform.PassContext(opt_level=3, config={}):
        lib = relay.build(mod, target=target, params=params)

dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))

################################################################################
# Comparing the Tuned and Untuned Models

meter.start()
raw_results = timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number)
meter.stop()

# get the energy consumption
trace = meter.get_trace()
sample = trace[0] # only one sample covering the entire period, as no "hotspots" were specified
# I take the package domain and subtract integrated graphics, the add DRAM as well. You can refer to the following diagram for explanation:
# https://pyjoules.readthedocs.io/en/latest/devices/intel_cpu.html#domains
total_energy_uJ = (sample.energy['package_0'] - sample.energy['uncore_0'] + sample.energy['dram_0'])
total_energy_J = total_energy_uJ / 1e6
average_energy_per_inference_J = total_energy_J / timing_number

optimized_times_s = (
    np.array(raw_results)
    / timing_number
)
optimized_flops = total_flop_count / optimized_times_s
optimized_watts = average_energy_per_inference_J / optimized_times_s
optimized_flops_per_watt = optimized_flops / optimized_watts
optimized_gflops_per_watt = optimized_flops_per_watt / 1e9  # convert to gigaflops

optimized_stats_seconds = {
    "mean": np.mean(optimized_times_s),
    "median": np.median(optimized_times_s),
    "std": np.std(optimized_times_s),
}
optimized_stats_gflops_per_watt = {
    "mean": np.mean(optimized_gflops_per_watt),
    "median": np.median(optimized_gflops_per_watt),
    "std": np.std(optimized_gflops_per_watt),
}
# ointervalintervaltime in milliseconds!
print("optimized time: %s" % (optimized_stats_seconds))
print("optimized gflops/watt: %s" % (optimized_stats_gflops_per_watt))
