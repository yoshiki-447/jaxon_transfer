#!/bin/bash

TIMESTAMP=$(date "+%Y%m%d_%H%M%S")

OUTPUT_DIR=~/ros/jaxon_transfer/rosbag
OUTPUT_FILE="rosbag_$TIMESTAMP.bag"

TOPICS="/joint_states"

rosbag record -O $OUTPUT_DIR/$OUTPUT_FILE $TOPICS 
