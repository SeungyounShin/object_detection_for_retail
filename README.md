# object detection for retail
an autonomous solution for unmanned shops based on computer vision

## object detection model
we use centernet( [paper](https://arxiv.org/pdf/1904.08189.pdf)  ) for detecting objects
> path  : ./nets

> train : train.py
* input  : images (512x384)
* output : heatmap, regression offsets, sizes, theta

why theta?
+ get more accurate RoI(region of interest)
+ which leads to get more robust feature (improve performance of recognition)
+ For these reasons, rbox gives kind of alignment effect

**model diagram**

![centernet](https://github.com/SeungyounShin/object_detection_for_retail/blob/master/resource/centernetRot.png?raw=true)

A detection result with rotated bounding box
![test](https://github.com/SeungyounShin/object_detection_for_retail/blob/master/resource/test.png?raw=true)

## recognition model
> ./recognition

further development will be tested with vargfacenet( [paper](https://arxiv.org/abs/1910.04985) )
currently we use pretrained-efficientent

**model diagram**

![arcface](https://github.com/SeungyounShin/object_detection_for_retail/blob/master/resource/arcface_infer.png?raw=true)

**our approach**
1. we get patches from inference model(object detection model) output
2. padding the images for purpose of handdling dimension error and accurate result
![patches](https://github.com/SeungyounShin/object_detection_for_retail/blob/master/resource/patches.png?raw=true)

## face analysis
we use azure cognitive service to detect 
+ faces in images
+ face age, gender
+ identification

## dependency
+ pytorch 1.1.0
+ numpy 1.18.1
+ pillow 7.0.0
