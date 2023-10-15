#!/bin/bash

if [[ $1 == clean ]]; then
	cd src/circulation/scripts
	make clean
	cd ../../..
fi

if [[ $1 == init ]] || [[ $2 == init ]] || [[ $3 == init ]]; then
	python3 -m pip install -r requirements.txt
	if [ ! -d src/geometry2 ]; then
		wstool init
		if [ `rosversion -d` == melodic ]; then
			wstool set -y src/geometry2 --git https://github.com/ros/geometry2 -v 0.6.7
		else
			wstool set -y src/geometry2 --git https://github.com/ros/geometry2 -v 0.7.6
		fi
		wstool update
	fi
	rosdep install --from-paths src --ignore-src -y -r
fi

if [[ $1 == build ]] || [[ $2 == build ]] || [[ $3 == build ]]; then
	if [ `rosversion -d` == melodic ]; then
		catkin build -j4 -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
	else
		catkin build -j4
	fi
fi

if [[ $1 == cython ]] || [[ $2 == cython ]] || [[ $3 == cython ]]; then
	cd src/circulation/scripts
	make all
	#cythonize -3 -a -i trajeometry.pyx
	#cythonize -3 -a -i trajutil.pyx
	#cythonize -3 -a -i fish2bird.pyx
	#cythonize -3 -a -i linetrack.pyx
	#cythonize -3 -a -i fuzzylines.pyx
	#cythonize -3 -a -i positioning.pyx
	#cythonize -3 -a -i trajectorybuild.pyx
	cd ../../..

	cp src/circulation/scripts/trajutil*.so devel/lib/python3/dist-packages/
	cp src/circulation/scripts/trajeometry*.so devel/lib/python3/dist-packages/
	cp src/circulation/scripts/fish2bird*.so devel/lib/python3/dist-packages/
	cp src/circulation/scripts/linetrack*.so devel/lib/python3/dist-packages/
	cp src/circulation/scripts/fuzzylines*.so devel/lib/python3/dist-packages/
	cp src/circulation/scripts/trajectorybuild*.so devel/lib/python3/dist-packages/
	cp src/trafficsigns/scripts/traffic_sign_detection.py devel/lib/python3/dist-packages/
fi

if [[ $1 == pack ]] || [[ $2 == pack ]] || [[ $3 == pack ]] || [[ $4 == pack ]]; then
	mkdir -p _tmp_circulationpack/src/circulation/scripts
	cd _tmp_circulationpack
	cp -r ../src/circulation/srv src/circulation/srv
	cp -r ../src/circulation/msg src/circulation/msg
	cp ../src/circulation/scripts/*.pyx src/circulation/scripts
	cp ../src/circulation/scripts/*.pxd src/circulation/scripts
	cp ../src/circulation/scripts/*.hpp src/circulation/scripts
	cp ../src/circulation/scripts/*.py src/circulation/scripts
	cp ../src/circulation/package.xml src/circulation/package.xml
	cp ../src/circulation/CMakeLists.txt src/circulation/CMakeLists.txt
	cp ../build_circulation.sh build_circulation.sh
	cp ../circulation4.yml circulation4.yml
	cp ../circulation4-2.yml circulation4-2.yml
	cp ../road_network.json road_network.json
	cp ../requirements.txt requirements.txt
	tar -czvf ../circulation_package-`date +%Y%m%d-%H%M%S`.tar.gz *
	cd ..
	rm -rf _tmp_circulationpack
fi