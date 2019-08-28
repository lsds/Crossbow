# Crossbow: A Multi-GPU Deep Learning System for Training with Small Batch Sizes

**Crossbow** is a multi-GPU system for training deep learning models that
  allows users to choose freely their preferred batch size, however small,
  while scaling to multiple GPUs. 
  
**Crossbow** utilises modern GPUs better than other systems by training multiple  _model replicas_ on the same GPU. When the batch size is sufficiently small to leave GPU resources unused, **Crossbow** trains a second model replica, a third, etc., as long as training throughput increases.

To synchronise many model replicas, **Crossbow** uses _synchronous model averaging_ to adjust the trajectory of each individual replica based on the average of all. With model averaging, the batch size does not increase linearly with the number of model replicas, as it would with synchronous SGD. This yields better statistical efficiency without cumbersome hyper-parameter tuning when trying to scale training to a larger number of GPUs.

See our [VLDB 2019 paper](http://www.vldb.org/pvldb/vol12/p1399-koliousis.pdf) for more details.

The system supports a variety of training algorithms, including synchronous SGD. We are working to seemlesly port existing TensorFlow models to Crossbow. 

## Installing Crossbow

### Prerequisites

**Crossbow** has been primarily tested on Ubuntu Linux 16.04. It requires the following Linux packages:

```shell
$ sudo apt-get install build-essential git openjdk-8-jdk maven libboost-all-dev graphviz wget
```
 
**Crossbow** requires NVIDIA's [CUDA](https://developer.nvidia.com/cuda-toolkit) toolkit, the [cuDDN](https://developer.nvidia.com/cudnn) library and the [NCCL](https://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html) library (currently using versions 8.0, 6.0, and 2.1.15, respectively). After successful installation, make sure that:

* `CUDA_HOME` is set (the default location is `/usr/local/cuda`)
* `NCCL_HOME` is set

and that:

* `PATH` includes `$CUDA_HOME/bin` and
* `LD_LIBRARY_PATH` includes `$CUDA_HOME/lib64` and `$NCCL_HOME/lib`

**Crossbow** also requires the [OpenBLAS](https://github.com/xianyi/OpenBLAS.git) and [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo) libraries. After successful installation, make sure that:

* `BLAS_HOME` is set (the default location is `/opt/OpenBLAS`)
* `JPEG_HOME` is set

and that:

* `LD_LIBRARY_PATH` includes `$BLAS_HOME/lib` and `$JPEG_HOME/lib`

### Configure OS

**Crossbow** uses page-locked memory regions to speed up data transfers from CPU to GPU and vice versa. The amount of memory locked by the system usually exceeds the default OS limit. Edit `/etc/security/limits.conf` and append the following lines to the end of the file:

```
*	hard	memlock	unlimited
* 	soft	memlock	unlimited
```

Save changes and reboot the machine.

### Building Crossbow

Assuming all enviromental variables have been set, build Crossbow's Java and C/C++ library:

```shell
$ git clone http://github.com/lsds/Crossbow.git
$ cd Crossbow
$ export CROSSBOW_HOME=`pwd`
$ ./scripts/build.sh
```

_**Note:** We will shortly add an installation script as well as a Docker image to simplify the installation process and avoid library conflicts._

## Training one of our benchmark models

### ResNet-50

**Crossbow** serialises [ImageNet](http://www.image-net.org) images and their labels into a binary format similar to TensorFlow's TFRecord. Follow [TensorFlow's instructions](https://github.com/tensorflow/models/blob/master/research/inception/README.md#getting-started) to download and convert the dataset to TFRecord format. You will end up with 1,024 training and 128 validation record files in a directory of your choice (say, `/data/imagenet/tfrecords`). Then, run:

```shell
$ cd $CROSSBOW_HOME
$ ./scripts/datasets/imagenet/prepare-imagenet.sh /data/imagenet/tfrecords /data/imagenet/crossbow
```

The script  will convert TensorFlow's record files to Crossbow's own binary format and store them in `/data/imagenet/crossbow`. You are now ready to train ResNet-50 with the ImageNet data set:

```shell
$ ./scripts/benchmarks/resnet-50.sh
```

### LeNet

The first script downloads the [MNIST](http://yann.lecun.com/exdb/mnist/) data set and converts it to Crossbow's binary record format. Output files are written in `$CROSSBOW_HOME/data/mnist/b-001` and they are tailored to a specific batch size (in this case, 1). The second script will train LeNet with the  MNIST data set.

```shell
$ cd $CROSSBOW_HOME
$ ./scripts/datasets/mnist/prepare-mnist.sh
$ ./scripts/benchmarks/lenet.sh
```

### Others

**Crossbow** supports the entire ResNet family of neural networks. It also supports VGG-16 based on the implementation [here](https://github.com/geifmany/cifar-vgg). It supports the [convnet-benchmarks](https://github.com/soumith/convnet-benchmarks) suite of micro-benchmarks too.

_**Note:** We will shortly add a page describing how to configure Crossbow's system parameters._

## Trying your first Crossbow program

**Crossbow** represents a deep learning application as a data flow graph: nodes
  represent operations and edges the data (multi-dimensional arrays, also known
  as _tensors_) that flow among them. The most notable operators are
  inner-product, pooling, convolutional layers and activation functions. Some of these operators have _learnable_ parameters (also multi-dimensional arrays) that form part of the model being trained. An inner-product operator, for example, has two learnable parameters, `weights` and `bias`:

```java
InnerProductConf conf = new InnerProductConf ();

/* Let's assume that there are 10 possible output labels, as in MNIST */
conf.setNumberOfOutputs (10);

/* Initialise weights with values drawn a random Gaussian distribution; 
 * and all of bias elements with the same value */
conf.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.GAUSSIAN).setStd(0.1F));
conf.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue(1F));

/* Create inner-product operator and wrap it in a graph node */
Operator op = new Operator ("InnerProduct", new InnerProduct (conf));
DataflowNode innerproduct = new DataflowNode (op);
```

Connect data flow nodes together to form a neural network. For example, we can connect the forward layers of a logistic regression model:

```java
innerproduct.connectTo(softmax).connectTo(loss);
```

At the end, we can construct our model and train it for 1 epoch:

```java
SubGraph subgraph = new SubGraph (innerproduct);
Dataflow dataflow = new Dataflow (subgraph).setPhase(Phase.TRAIN);
ExecutionContext context = new ExecutionContext (new Dataflow [] { dataflow, null });
context.init();
context.train(1, TrainingUnit.EPOCHS);
```

The full source code is available [here](src/test/java/uk/ac/imperial/lsds/crossbow/LogisticRegression.java).

## For more information

* [LSDS Website](https://www.lsds.doc.ic.ac.uk) 

## Licence

[Apache License 2.0](LICENSE)
