## Build

Using the following command to build a custom image for CrossBow based on CUDA 9.2.

```bash
$ docker build -f ./dockerfiles/crossbow-cuda-9.2.Dockerfile -t crossbow:latest .
```

## LeNet example

We run a LeNet example under the: ./crossbow/scripts/benchmarks/lenet.sh.

Here we assume that the data disk is under: /home/akolious/.m2/:/root/.m2. You can mount your data disk by replacing this path.

```bash
$ docker run --runtime=nvidia -u root:root -v /home/akolious/.m2/:/root/.m2 --ulimit memlock=1073741824:1073741824 -it crossbow:latest ./crossbow/scripts/benchmarks/lenet.sh
```

Current issues:

TODO:

```
debconf: unable to initialize frontend: Dialog
debconf: (TERM is not set, so the dialog frontend is not usable.)
debconf: falling back to frontend: Readline
debconf: unable to initialize frontend: Readline
debconf: (This frontend requires a controlling tty.)
debconf: falling back to frontend: Teletype
dpkg-preconfigure: unable to re-open stdin:
```

TODO:

```
cudnn/cudnnhelper.c: In function ‘cudnnActivationModeString’:
cudnn/cudnnhelper.c:5:2: warning: enumeration value ‘CUDNN_ACTIVATION_IDENTITY’ not handled in switch [-Wswitch]
  switch (mode) {
  ^
```
