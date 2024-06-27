(arm-solve-ik :rarm-trans-pos #f(100 0 300))
(send *ri* :angle-vector (send *robot* :angle-vector))
(send *ri* :wait-polation)

(arm-solve-ik :rarm-trans-pos #f(100 100 100))
(send *ri* :angle-vector (send *robot* :angle-vector))
(send *ri* :wait-polation)

(arm-solve-ik :rarm-trans-pos #f(0 0 100))
(send *ri* :angle-vector (send *robot* :angle-vector))
(send *ri* :wait-polation)

(arm-solve-ik :rarm-trans-pos #f(0 100 0))
(send *ri* :angle-vector (send *robot* :angle-vector))
(send *ri* :wait-polation)

(arm-solve-ik :rarm-trans-pos #f(100 0 0))
(send *ri* :angle-vector (send *robot* :angle-vector))
(send *ri* :wait-polation)

(arm-solve-ik :rarm-trans-pos #f(0 100 50))
(send *ri* :angle-vector (send *robot* :angle-vector))
(send *ri* :wait-polation)

(arm-solve-ik :larm-trans-pos #f(0 100 -50))
(send *ri* :angle-vector (send *robot* :angle-vector))
(send *ri* :wait-polation)

(send *robot* :larm :wrist-y :joint-angle 0)
(send *ri* :angle-vector (send *robot* :angle-vector))
(send *ri* :wait-polation)

(arm-solve-ik :larm-trans-pos #f(50 100 0))
(send *ri* :angle-vector (send *robot* :angle-vector))
(send *ri* :wait-polation)

