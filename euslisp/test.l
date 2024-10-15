(load "package://pr2eus/pr2.l")
(require "models/arrow-object.l")

(send *robot* :reset-pose)
(send *robot* :rarm-wrist-r :joint-angle 0)
(send *robot* :rarm-wrist-p :joint-angle 0)
(send *robot* :rarm-wrist-y :joint-angle 0)
(send *robot* :larm-wrist-r :joint-angle 0)
(send *robot* :larm-wrist-p :joint-angle 0)
(send *robot* :larm-wrist-y :joint-angle 0)
(send *robot* :rarm :move-end-pos #f(-80 0 80) :world)
(send *robot* :larm :move-end-pos #f(-80 0 80) :world)

(setq rarm_link5 (send *robot* :rarm_link5_lk :copy-worldcoords))
(format t "end-coords: ~A~%" (send *robot* :rarm :end-coords :copy-worldcoords))
(format t "before rarm-link5: ~A~%" rarm_link5)

(setq move-target-coords (arrow))
(send move-target-coords :newcoords (send rarm_link5 :rotate (/ pi 2.0) :y))

(send (send (send *robot* :rarm :end-coords) :parent) :assoc move-target-coords)
(setq target-coords (send (send (send *robot* :rarm :end-coords :copy-worldcoords)
			  :translate #f(-100 0 100))
			  :rotate (/ pi 2.0) :x))
(send *robot* :rarm :inverse-kinematics target-coords
      :move-target move-target-coords)

(setq rarm_link5 (send *robot* :rarm_link5_lk :copy-worldcoords))
(format t "after rarm-link5: ~A~%" rarm_link5)

(objects (list *robot* rarm_link5 move-target-coords target-coords))
