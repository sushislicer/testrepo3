### Configure GLU Library

`sudo apt-get update && sudo apt-get install -y libglu1-mesa-dev libgl1-mesa-dev`

### Configure EGL for Ubuntu 24.04

`sudo apt-get update && sudo apt-get install -y libegl1 libgl1 libglib2.0-0t64`

Then, edit your .bashrc file and add the following line:

`export PYOPENGL_PLATFORM=egl`
`export PYRENDER_PLATFORM=egl`
`export XDG_RUNTIME_DIR=/tmp/xdg` 
`export EGL_PLATFORM=surfaceless` 
`export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json `

Then, source your .bashrc file:

`source ~/.bashrc`

Then, test if EGL is working:

`python -c "import pyrender; r = pyrender.OffscreenRenderer(100, 100); print('EGL ready: OK')"`
`XDG_RUNTIME_DIR=/tmp/xdg EGL_PLATFORM=surfaceless __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json python -c "from open3d.visualization import rendering; r=rendering.OffscreenRenderer(64,64); r.release(); print('OK')"`



Please fix "the semantic channel is effectively dead because its input render is not a coherent 2D depiction of the object", and "