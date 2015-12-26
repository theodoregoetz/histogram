#!/bin/bash

if [[ -d test_images ]]; then
	rm test_images/*.*
	rmdir test_images
fi

nosetests --with-coverage --cover-inclusive --cover-package=histogram

for i in master_images/*.png; do
    perceptualdiff -fov 85 -threshold 10 -output $i.ppm $i ${i/master/test}
done
