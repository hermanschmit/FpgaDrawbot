#!/bin/bash
for szFile in $1/fig?????.png
do convert $szFile -rotate 90 $1/$(basename -s .png $szFile)_rot.png
done

ffmpeg -y -framerate 10 -i $1/fig%05d_rot.png -c:v libx264 -r 30 -pix_fmt yuv420p $1/out.mp4
