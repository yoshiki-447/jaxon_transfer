#!/bin/bash

SOURCE_SCRIPT="source ~/ros/auto_ws/devel/setup.bash"

TIMESTAMP=$(date "+%Y%m%d_%H%M%S")

OUTPUT_DIR=$(rospack find jaxon_transfer)/rosbag
OUTPUT_FILE="rosbag_$TIMESTAMP.bag"

TOPICS="/joint_states /tf /tf_static /off_rhsensor /off_lhsensor /clock /ref_rh_wrench /ref_lh_wrench"

rosbag record -O $OUTPUT_DIR/$OUTPUT_FILE $TOPICS 
