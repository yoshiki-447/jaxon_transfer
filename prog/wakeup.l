(send *ri* :set-foot-steps
      (list (make-coords (send *robot* :lleg :end-coords :copy-worldcoords) :name :lleg)
	    (make-coords (send
			  (send *robot* :rleg :end-coords :copy-worldcoords)
			  :translate (float-vector 100 -200 0))
			 :name :rleg)
	    )
      )

(lower-waist)
(lower-waist)

(zmp-start)
(arm-solve-ik :larm-trans-pos #f(300 0 200) :rarm-trans-pos #f(100 -100 -100))
(send *ri* :angle-vector (send *robot* :angle-vector))
(send *ri* :wait-polation)

(arm-solve-ik :larm-trans-pos #f(100 0 0) :rarm-trans-pos #f(100 -100 -100))
(send *ri* :angle-vector (send *robot* :angle-vector))
(send *ri* :wait-polation)

(arm-solve-ik :rarm-trans-pos #f(100 0 0))
(send *ri* :angle-vector (send *robot* :angle-vector))
(send *ri* :wait-polation)

(arm-solve-ik :larm-trans-pos #f(200 0 0))
(send *ri* :angle-vector (send *robot* :angle-vector))
(send *ri* :wait-polation)

(send *robot* :rarm :wrist-r :joint-angle 90)
(send *robot* :larm :wrist-y :joint-angle 0)
(send *ri* :angle-vector (send *robot* :angle-vector))
(send *ri* :wait-polation)
