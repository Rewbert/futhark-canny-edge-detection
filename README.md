## Canny Edge Detection in Futhark
This repository contains some very basic image processing methods, in particular an implementation of Canny Edge Detection, implemented in Futhark.

There is also an example illustrating how these simple methods can be used to implement a lane detection algorithm for use in autonomous operation or something similar.

### Running
You will need a couple of things to run the code:
+ [Futhark](http://futhark.readthedocs.io/en/latest/).
+ openCL
+ pyopencl (i suppose the easiest is `apt-get install python-pyopencl`
+ sklearn / scikit-learn
+ openCV
+ Other things i most certainly forgot to mention here.
+ some video, preferably a dark road with very white lines. You need to change the line in lane_detection.py that specifies which input file you have (line 56).

To compile and execute, you need to run these commands.
```
futhark-pyopencl --library util.fut
```
```
futhark-pyopencl --library imageproc.fut
```
```
python lane_detection.py
```

Hopefully, you will get a result like shown below. If not, look into tweaking the argument for canny.
![image broken](https://github.com/Rewbert/futhark-canny-edge-detection/blob/master/images/lane-det.png)

### Performance
My potato-laptop has:
+ Intel Core i3-6100U Processor (3MB cache, 2.30GHz)
+ Intel HD Graphics 520 (it's an integrated card)
+ 4GB DDR4-2133 SODIMM RAM

The performance of the Canny Edge Detection algorithm is (evaluated with futhark-bench):
+ HD resolution images 1920x1080:
    + 272 916 microseconds (running on CPU)
    + 99 843 microseconds (running on GPU) (2.73x faster)
+ 4K HD resolution images 3840x2160:
    + 1 094 854 microseconds (running on CPU)
    + 387 495 microseconds (running on GPU) (2.82x faster)


+ with prettier convolution:
+ HD resolution images 1920x1080:
    + 314 348 microseconds (running on CPU)
    + 117 152 microseconds (running on GPU)
+ 4K HD resolution images 3840x2160:
    + 1 254 759 microseconds (running on CPU)
    + 446 858 microseconds (running on GPU)
