echo "Creating a test directory"
mkdir model && cd model

echo "Fetching Yolo V3 CFG"
wget https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg

echo "Fetching Yolo V3 Weights"
wget https://pjreddie.com/media/files/yolov3.weights
