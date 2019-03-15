## Build

The following command builds a custom image for Crossbow based on CUDA 9.2.

```bash
$ docker build -f ./dockerfiles/crossbow-cuda-9.2.Dockerfile -t crossbow:latest .
```

## Run

Run the LeNet example with the following command:

```bash
$ docker run --runtime=nvidia -u root:root --ulimit memlock=1073741824:1073741824 -it crossbow:latest /crossbow/scripts/benchmarks/lenet.sh
```

