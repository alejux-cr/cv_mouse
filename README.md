# Headmouse - Computer Vision application to control the mouse by tracking face/head movement

## Computer Vision - ENV set up on Ubuntu 20.04 Focal Fossa
All steps below are specifically for Linux distro Ubuntu 20.04 LTS

## System requirements
- Python 3.8
- Pip 20.2.4
- OpenCV

### Setup
https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/

The recommended approach is setting up a virtual env with venv as stated in the official python docs.

```sh
$ python3 -m pip install --user --upgrade pip
$ apt-get install python3-venv
```

Create a virtual env

```sh
$ python3 -m venv path/env
```

Activating the virtual env

```sh
$ source path/env/bin/activate
```

Confirm you're in the virtual env

```sh
$ which python
.../path/env/bin/python
```

Leaving the virtual env


### Install OpenCV contrib package via pip inside virtual env
```sh
$ pip install opencv-contrib-python
```


