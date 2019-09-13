# Hand Tracker
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/wolterlw/hand_tracking/master)

Simple Python wrapper for Google's [Mediapipe Hand Tracking](https://github.com/google/mediapipe/blob/master/mediapipe/docs/hand_tracking_mobile_gpu.md) pipeline.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Requirements

These are required to use the HandTracker module

```
numpy
opencv
tensorflow (custom build)
```
To run the Jupyter demo you'll also need `jupyter` and `matplotlib`


### Setup 

To use the palm detection .tflite model we need to compile a custom [TensorFlow](https://github.com/tensorflow/tensorflow) build with a tflite interpreter wrapper with added CustomOperations whose implementations reside in [mediapipe/mediapipe/util/tflite/operations](https://github.com/google/mediapipe/tree/master/mediapipe/util/tflite/operations).
(TensorFlow version used [1.13](https://github.com/tensorflow/tensorflow/releases/tag/v1.13.2)).

Move files inside this folder to a convenient location within TensorFlow repository. I used `tensorflow/tensorflow/lite/python/custom_ops`.
Now we need to change the BUILD file. 
Change package visibility to `["//visibility:public"]`, as we now are inside tensorflow repo you should remove the `@org_tensorflow` from dependency paths and that's it. 
To build this version of TensorFlow you'll need [Bazel 0.21.0](https://github.com/bazelbuild/bazel/releases/tag/0.21.0). Setting up a new virtual environment to install this custom build would also be a good idea. (I've used conda for that)

Proceed with bazel workspace configuration as per TensorFlow compilation guide. As TFLite does not support desktop GPUs as of the time of this experiment you can save yourself some trouble by selecting [n] to CUDA support.

To start building call 
```bazel build --action_env PATH="$PATH" --noincompatible_strict_action_env --config=opt --incompatible_disable_deprecated_attr_params=false //tensorflow/tools/pip_package:build_pip_package```
Additional parameters make sure bazel works well with the virtual env.
After you've successfully executed the build command you end up with a binary `./bazel-bin/tensorflow/tools/pip_package/build_pip_package`, which you need to execute providing a valid path to store the python wheel as the only parameter. After the wheel is built make sure you're in the right virtual environment and install it.

Congrats, now you know how to clumsily add custom operations to tflite.

Alternatively here's a [link](https://www.dropbox.com/s/07p84k7q4kxwc02/tensorflow-1.13.2-cp37-cp37m-linux_x86_64.whl?dl=0) to a wheel built by me on a Ubuntu 18.04 laptop.

## Acknowledgments

This work is a study of models developed by Google and distributed as a part of the [Mediapipe](https://github.com/google/mediapipe) framework.
