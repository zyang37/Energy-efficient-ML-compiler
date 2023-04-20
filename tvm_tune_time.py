# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Compiling and Optimizing a Model with the Python Interface (AutoTVM)
====================================================================
**Author**:
`Chris Hoge <https://github.com/hogepodge>`_

In the `TVMC Tutorial <tvmc_command_line_driver>`_, we covered how to compile, run, and tune a
pre-trained vision model, ResNet-18 v2 using the command line interface for
TVM, TVMC. TVM is more that just a command-line tool though, it is an
optimizing framework with APIs available for a number of different languages
that gives you tremendous flexibility in working with machine learning models.

In this tutorial we will cover the same ground we did with TVMC, but show how
it is done with the Python API. Upon completion of this section, we will have
used the Python API for TVM to accomplish the following tasks:

* Compile a pre-trained ResNet-18 v2 model for the TVM runtime.
* Run a real image through the compiled model, and interpret the output and model
  performance.
* Tune the model that model on a CPU using TVM.
* Re-compile an optimized model using the tuning data collected by TVM.
* Run the image through the optimized model, and compare the output and model
  performance.

The goal of this section is to give you an overview of TVM's capabilites and
how to use them through the Python API.
"""

# pick your flavor of resnet<XX>-v2-7.onnx
# https://github.com/onnx/models/tree/main/vision/classification/resnet/model
RESNET = 18 

################################################################################
# TVM is a deep learning compiler framework, with a number of different modules
# available for working with deep learning models and operators. In this
# tutorial we will work through how to load, compile, and optimize a model
# using the Python API.
#
# We begin by importing a number of dependencies, including ``onnx`` for
# loading and converting the model, helper utilities for downloading test data,
# the Python Image Library for working with the image data, ``numpy`` for pre
# and post-processing of the image data, the TVM Relay framework, and the TVM
# Graph Executor.

import onnx
from tvm.contrib.download import download_testdata
from PIL import Image
import numpy as np
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor

################################################################################
# Downloading and Loading the ONNX Model
# --------------------------------------
#
# For this tutorial, we will be working with ResNet-18 v2. ResNet-18 is a
# convolutional neural network that is 18 layers deep and designed to classify
# images. The model we will be using has been pre-trained on more than a
# million images with 1000 different classifications. The network has an input
# image size of 224x224. If you are interested exploring more of how the
# ResNet-18 model is structured, we recommend downloading
# `Netron <https://netron.app>`_, a freely available ML model viewer.
#
# TVM provides a helper library to download pre-trained models. By providing a
# model URL, file name, and model type through the module, TVM will download
# the model and save it to disk. For the instance of an ONNX model, you can
# then load it into memory using the ONNX runtime.
#
# .. admonition:: Working with Other Model Formats
#
#   TVM supports many popular model formats. A list can be found in the
#   :ref:`Compile Deep Learning Models <tutorial-frontend>` section of the TVM
#   Documentation.

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
# ------------------------------------------------------
#
# Each model is particular when it comes to expected tensor shapes, formats and
# data types. For this reason, most models require some pre and
# post-processing, to ensure the input is valid and to interpret the output.
# TVMC has adopted NumPy's ``.npz`` format for both input and output data.
#
# As input for this tutorial, we will use the image of a cat, but you can feel
# free to substitute this image for any of your choosing.
#
# .. image:: https://s3.amazonaws.com/model-server/inputs/kitten.jpg
#    :height: 224px
#    :width: 224px
#    :align: center
#
# Download the image data, then convert it to a numpy array to use as an input to the model.

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
# ----------------------------
#
# The next step is to compile the ResNet model. We begin by importing the model
# to relay using the `from_onnx` importer. We then build the model, with
# standard optimizations, into a TVM library.  Finally, we create a TVM graph
# runtime module from the library.

target = "llvm -mcpu=skylake"

######################################################################
# .. admonition:: Defining the Correct Target
#
#   Specifying the correct target can have a huge impact on the performance of
#   the compiled module, as it can take advantage of hardware features
#   available on the target. For more information, please refer to
#   :ref:`Auto-tuning a convolutional network for x86 CPU <tune_relay_x86>`.
#   We recommend identifying which CPU you are running, along with optional
#   features, and set the target appropriately. For example, for some
#   processors ``target = "llvm -mcpu=skylake"``, or ``target = "llvm
#   -mcpu=skylake-avx512"`` for processors with the AVX-512 vector instruction
#   set.
#

# The input name may vary across model types. You can use a tool
# like Netron to check input names
input_name = "data"
shape_dict = {input_name: img_data.shape}

mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))

######################################################################
# Execute on the TVM Runtime
# --------------------------
# Now that we've compiled the model, we can use the TVM runtime to make
# predictions with it. To use TVM to run the model and make predictions, we
# need two things:
#
# - The compiled model, which we just produced.
# - Valid input to the model to make predictions on.

dtype = "float32"
module.set_input(input_name, img_data)
module.run()
output_shape = (1, 1000)
tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()

######################################################################
# Count the total model of floating point operations within this model
from tvm import autotvm

total_flop_count = 0
tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)
for i, task in enumerate(tasks):
    total_flop_count += task.flop


################################################################################
# Tune the model
# --------------
# The previous model was compiled to work on the TVM runtime, but did not
# include any platform specific optimization. In this section, we will show you
# how to build an optimized model using TVM to target your working platform.
#
# In some cases, we might not get the expected performance when running
# inferences using our compiled module. In cases like this, we can make use of
# the auto-tuner, to find a better configuration for our model and get a boost
# in performance. Tuning in TVM refers to the process by which a model is
# optimized to run faster on a given target. This differs from training or
# fine-tuning in that it does not affect the accuracy of the model, but only
# the runtime performance. As part of the tuning process, TVM will try running
# many different operator implementation variants to see which perform best.
# The results of these runs are stored in a tuning records file.
#
# In the simplest form, tuning requires you to provide three things:
#
# - the target specification of the device you intend to run this model on
# - the path to an output file in which the tuning records will be stored
# - a path to the model to be tuned.
#

import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner
from tvm import autotvm

################################################################################
# Set up some basic parameters for the runner. The runner takes compiled code
# that is generated with a specific set of parameters and measures the
# performance of it. ``number`` specifies the number of different
# configurations that we will test, while ``repeat`` specifies how many
# measurements we will take of each configuration. ``min_repeat_ms`` is a value
# that specifies how long need to run configuration test. If the number of
# repeats falls under this time, it will be increased. This option is necessary
# for accurate tuning on GPUs, and is not required for CPU tuning. Setting this
# value to 0 disables it. The ``timeout`` places an upper limit on how long to
# run training code for each tested configuration.

number = 10 # 250  # only test a single configuration at a time
repeat = 100  # each configuration is tested X times. Keep in mind that for psutil, they reccomend a measure duration of at least 0.1s, so you should run the model at least 0.1s for measuring to be *somewhat* accurate
min_repeat_ms = 10  # since we're tuning on a CPU, can be set to 0
timeout = 20  # in seconds

# create a TVM runner
runner = autotvm.LocalRunner(
    number=number,
    repeat=repeat,
    timeout=10,  # in seconds
    min_repeat_ms=min_repeat_ms,
    enable_cpu_cache_flush=True,
)

################################################################################
# Create a simple structure for holding tuning options. We use an XGBoost
# algorithim for guiding the search. For a production job, you will want to set
# the number of trials to be larger than the value of 20 used here. For CPU we
# recommend 1500, for GPU 3000-4000. The number of trials required can depend
# on the particular model and processor, so it's worth spending some time
# evaluating performance across a range of values to find the best balance
# between tuning time and model optimization. Because running tuning is time
# intensive we set number of trials to 10, but do not recommend a value this
# small. The ``early_stopping`` parameter is the minimum number of trails to
# run before a condition that stops the search early can be applied. The
# measure option indicates where trial code will be built, and where it will be
# run. In this case, we're using the ``LocalRunner`` we just created and a
# ``LocalBuilder``. The ``tuning_records`` option specifies a file to write
# the tuning data to.

tuning_option = {
    "tuner": "xgb",
    "trials": 1500,
    "early_stopping": 100,
    "measure_option": autotvm.measure_option(
        # local runner has n_parallel=1 hardcoded by default
        builder=autotvm.LocalBuilder(build_func="default", n_parallel=1), runner=runner
    ),
    "tuning_records": f"resnet-{RESNET}-v2-autotuning_with_time.json",
}

################################################################################
# .. admonition:: Defining the Tuning Search Algorithm
#
#   By default this search is guided using an `XGBoost Grid` algorithm.
#   Depending on your model complexity and amount of time available, you might
#   want to choose a different algorithm.


################################################################################
# .. admonition:: Setting Tuning Parameters
#
#   In this example, in the interest of time, we set the number of trials and
#   early stopping to 20 and 100. You will likely see more performance improvements if
#   you set these values to be higher but this comes at the expense of time
#   spent tuning. The number of trials required for convergence will vary
#   depending on the specifics of the model and the target platform.

# begin by extracting the tasks from the onnx model
tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

# Tune the extracted tasks sequentially.
for i, task in enumerate(tasks):
    prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
    tuner_obj = XGBTuner(task, loss_type="rank", num_threads=1)
    tuner_obj.tune(
        n_trial=min(tuning_option["trials"], len(task.config_space)),
        early_stopping=tuning_option["early_stopping"],
        measure_option=tuning_option["measure_option"],
        callbacks=[
            autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
            autotvm.callback.log_to_file(tuning_option["tuning_records"]),
        ],
    )
