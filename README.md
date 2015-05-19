# Computer Vision exercises

These are some programs I'm developing while working through
[the Szeliski computer vision book](http://szeliski.org/Book/). They're
probably not of general interest unless you're looking for some cheesy
OpenCV examples written in C++.

## Building and running

    # Install dependencies (on Ubuntu):
    sudo apt-get install cmake libopencv-dev

    # Install dependencies (on OSX):
    brew tap homebrew/science
    brew install opencv

    # Clone sources
    cd somewhere
    git clone https://github.com/mmueller/cv.git

    # Make a build directory
    cd cv
    mkdir build

    # Build stuff
    cd build
    cmake ..
    make

    # Run stuff
    ./color_balance
