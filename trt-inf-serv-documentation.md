# TensorRT Inference Server
How do we serve a Tensorflow model using TensorRT Inference Server on Google Cloud?

What is TensorRT? From the [docs](https://developer.nvidia.com/tensorrt):
> NVIDIA TensorRT™ is a platform for high-performance deep learning inference. It includes a deep learning inference optimizer and runtime that delivers low latency and high-throughput for deep learning inference applications. TensorRT-based applications perform up to 40x faster than CPU-only platforms during inference. With TensorRT, you can optimize neural network models trained in all major frameworks, calibrate for lower precision with high accuracy, and finally deploy to hyperscale data centers, embedded, or automotive product platforms.

What is TensorRT Inference Server? From the [docs](https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-guide/docs/):
> The NVIDIA TensorRT Inference Server provides a cloud inferencing solution optimized for NVIDIA GPUs. The server provides an inference service via an HTTP or gRPC endpoint, allowing remote clients to request inferencing for any model being managed by the server.

The TensorRT Inference Server is available in two ways:

* As a pre-built Docker container available from the NVIDIA GPU Cloud (NGC)
* As buildable source code located in GitHub. Building the inference server yourself requires the usage of Docker and the TensorFlow and PyTorch containers available from NGC

We are going to go ahead with the pre-built Docker container way.

## Create Environment on GCP

To create instances that will be used for model preparation:

```
export IMAGE_FAMILY="tf-latest-gpu"
export ZONE="us-west1-b"
export INSTANCE_NAME="model-prep"
gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --image-family=$IMAGE_FAMILY \
        --machine-type=n1-standard-8 \
        --image-project=deeplearning-platform-release \
        --maintenance-policy=TERMINATE \
        --accelerator="type=nvidia-tesla-t4,count=1" \
        --metadata="install-nvidia-driver=True"

```

## Prerequistes:
* Ensure you have access and are logged into NGC. For step-by-step instructions, see the [NGC Getting Started Guide](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html).


## Using a Pre-Built Docker Container:
* Use `docker pull` to get the TensorRT Inference Server container from NGC:
```
docker pull nvcr.io/nvidia/tensorrtserver:<xx.yy>-py3
```
Where <xx.yy> is the version of the inference server that you want to pull. Once you have the container follow these steps to run the server and the example client applications.

## Model Repository
The TensorRT Inference Server accesses models from a locally accessible file path or from Google Cloud Storage. This path is specified when the server is started using the `--model-store` option.

For a locally accessible file-system the absolute path must be specified, for example, --model-store=/path/to/model/repository. For a model repository residing in Google Cloud Storage, the path must be prefixed with gs://, for example, --model-store=gs://bucket/path/to/model/repository.


An example of a typical model repository named layout is shown below: 

```
└── models
    ├── resnet152
    │   ├── 1
    │   │   └── model.savedmodel
    │   │       ├── saved_model.pb
    │   │       └── variables
    │   │           ├── variables.data-00000-of-00001
    │   │           └── variables.index
    │   ├── 2
    │   │   └── model.savedmodel
    │   │       ├── saved_model.pb
    │   │       └── variables
    │   ├── 3
    │   │   └── model.savedmodel
    │   │       ├── saved_model.pb
    │   │       └── variables
    │   ├── 4
    │   │   └── model.savedmodel
    │   │       ├── saved_model.pb
    │   │       └── variables
    │   └── config.pbtxt
    └── resnet50
        ├── 1
        │   └── model.savedmodel
        │       ├── saved_model.pb
        │       └── variables
        │           ├── variables.data-00000-of-00001
        │           └── variables.index
        ├── 2
        │   └── model.savedmodel
        │       ├── saved_model.pb
        │       └── variables
        ├── 3
        │   └── model.savedmodel
        │       ├── saved_model.pb
        │       └── variables
        ├── 4
        │   └── model.savedmodel
        │       ├── saved_model.pb
        │       └── variables
        └── config.pbtxt
```

Any number of models may be specified and the inference server will attempt to load all models into the CPU and GPU when the server starts.

The name of the model directory (resnet152 and resnet50 in the above example) must match the name of the model specified in the model configuration file, `config.pbtxt`. Each model directory must have at least one numeric subdirectory. Each of these subdirectories holds a version of the model with the version number corresponding to the directory name.

For TensorFlow SavedModel models the name of the directory model containing the saved model must be `model.savedmodel`.

## Model Configuration

Each model in a Model Repository must include a model configuration that provides required and optional information about the model. Typically, this configuration is provided in a config.pbtxt file. A minimal model configuration must specify name, platform, max_batch_size, input, and output.

The minimal configuration for our example is:

```
name: "resnet152"
platform: "tensorflow_savedmodel"
max_batch_size: 1
input {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 256, 192, 3 ]
  }
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 64, 48, 17 ]
  }
]
version_policy: { specific { versions: 4 }}
```

The name of the model must match the name of the model repository directory containing the model. The platform must be one of tensorrt_plan, tensorflow_graphdef, tensorflow_savedmodel, caffe2_netdef, or custom.

For models that support batched inputs the max_batch_size value must be >= 1. The TensorRT Inference Server assumes that the batching occurs along a first dimension that is not listed in the inputs or outputs. For the above example, the server expects to receive input tensors with shape [ x, 256, 192, 3 ] and produces an output tensor with shape [ x, 64, 48, 17 ], where x is the batch size of the request.

The section on [Datatypes](https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-guide/docs/model_configuration.html#datatypes) describes the allowed datatypes and how they map to the datatypes of each model type.