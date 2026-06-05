#!/bin/bash

docker build -t checkerboard_container .
docker run -it -v "$(pwd):/work \
    -w /work" checkerboard_container