#!/bin/bash

source $(rospack find jaxon_ros_bridge)/scripts/upstart/byobu-utils.bash
SESSION_NAME=jaxon
create-session

SOURCE_SCRIPT="source ~/ros/auto_ws/devel/setup.bash"

new-window roseus "${SOURCE_SCRIPT} && rossetjaxon_red && cd $(rospack find jaxon_transfer)/euslisp && em -f shell"
new-window rviz "${SOURCE_SCRIPT} && rossetjaxon_red && rviz -d $(rospack find auto_stabilizer_config)/config/jaxonred-with-mslhand.rviz"
