;;;;finish-posture
send *robot* :rarm :collar-y :joint-angle 20
send *robot* :torso :waist-y :joint-angle 20
send *robot* :rarm :shoulder-y :joint-angle -5
send *robot* :rarm :shoulder-p :joint-angle -40
send *robot* :rarm :shoulder-r :joint-angle -20.0
send *robot* :rarm :elbow-p :joint-angle -60
send *robot* :rarm :wrist-p :joint-angle -20.0
send *robot* :rarm :wrist-r :joint-angle 40
send *robot* :rarm :wrist-y :joint-angle 0

send *robot* :larm :shoulder-p :joint-angle 10
send *robot* :larm :shoulder-y :joint-angle -10

;;;;start-posture
send *robot* :rarm :shoulder-p :joint-angle -20
send *robot* :rarm :elbow-p :joint-angle -60
send *robot* :rarm :wrist-r :joint-angle 40

send *robot* :larm :wrist-y :joint-angle -40
send *robot* :larm :shoulder-p :joint-angle 20
