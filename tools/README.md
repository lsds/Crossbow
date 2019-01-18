## Build

```shell
$ docker build -f ./dockerfiles/crossbow.Dockerfile -t cb .
```

## Run

```
docker run --runtime=nvidia -u root:root -v /home/akolious/.m2/:/root/.m2 --ulimit memlock=1073741824:1073741824 -it cb
```

Current issues:

TODO:

```
akolious@platypus2:~/crossbow.git/tools$ docker run --runtime=nvidia -u $(id -u):$(id -g) -v $(pwd):/my-devel -it cb
groups: cannot find name for group ID 1679
I have no name!@614d2413fd20:/$ ls
bin   dev  home  lib64  mnt   opt   root  sbin  sys  usr
boot  etc  lib   media  my-devel  proc  run   srv   tmp  var    
```

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
