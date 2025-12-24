### Configure GLU Library

`sudo apt-get update && sudo apt-get install -y libglu1-mesa-dev libgl1-mesa-dev`

### Configure EGL for Ubuntu 24.04

`sudo apt-get update && sudo apt-get install -y libegl1 libgl1 libglib2.0-0t64`

Then, edit your .bashrc file and add the following line:

`export PYOPENGL_PLATFORM=egl`
`export PYRENDER_PLATFORM=egl`

Then, source your .bashrc file:

`source ~/.bashrc`

Then, test if EGL is working:

`python -c "import pyrender; r = pyrender.OffscreenRenderer(100, 100); print('EGL ready: OK')"`