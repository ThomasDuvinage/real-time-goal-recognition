# Real time goal recognition 

This fork intends to simplify installation using Docker. 

## Docker installation 

After git clone the project, you can run the following command to build the image. 
This is only compatible with computer running with NVIDIA gpu. 

```bash
docker build -t real-time-goal-recognition .
```

Once it's built, you can run the run the image : 

```bash
docker run --gpus all -it --rm -e DISPLAY=${DISPLAY} -e NVIDIA_DRIVER_CAPABILITIES=all -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v $(pwd):/app real-time-goal-recognition
```

Once the docker is running, you need to source the conda environnement : 

```bash
source /opt/conda/bin/activate
```

Now you can run your script : 

```bash
python3 mainvideo.py
```

> Note that by default ZED SDK is installed but no other camera drivers. Please feel free to edit the `Dockerfile` to match your need.



