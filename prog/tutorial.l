;; (setq *rarm-end-coords* (send *robot* :rarm :end-coords :copy-worldcoords))
;; (setq *rarm-target-coords* (send *rarm-end-coords* :translate #f(100 100 0) :world))
;; (send *robot* :inverse-kinematics *rarm-target-coords*
;;       :move-target (send *robot* :rarm :end-coords) ;; 動かす対称
;;       :link-list (send *robot* :link-list (send *robot* :rarm :end-coords :parent))
;;       :translation-axis t
;;       :rotation-axis t
;;       )

(defun arm-solve-ik
    (&key(rarm-trans-pos #f(0 0 0))
	 (larm-trans-pos #f(0 0 0))
	 (rarm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad 0)))
	 (larm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad 0))))
  (let* ((rarm-end-coords (send *robot* :rarm :end-coords :copy-worldcoords))
	 (larm-end-coords (send *robot* :larm :end-coords :copy-worldcoords))
	 (rarm-translation-coords (send rarm-end-coords :translate rarm-trans-pos :world) :copy-worldcoords)
	 (larm-translation-coords (send larm-end-coords :translate larm-trans-pos :world) :copy-worldcoords)
	 (rarm-target (make-coords :pos (send rarm-translation-coords :pos)
				   :rpy rarm-rpy))
	 (larm-target (make-coords :pos (send larm-translation-coords :pos)
				   :rpy larm-rpy)))
    (send *robot* :inverse-kinematics (list rarm-target larm-target)
	  :move-target (list (send *robot* :rarm :end-coords)
			     (send *robot* :larm :end-coords))
	  :link-list (list (send *robot* :link-list (send *robot* :rarm :end-coords :parent))
			   (send *robot* :link-list (send *robot* :larm :end-coords :parent)))
	  :translation-axis (list t t)
	  :rotation-axis (list t t)
	  ;; :target-centroid-pos (midpoint 0.5
	  ;; 				 (send *robot* :rleg :end-coords :worldpos)
	  ;; 				 (send *robot* :rleg :end-coords :worldpos))
	  ;; :cog-translation-axis :z
	  )
    )
  )

(defun foot-solve-ik
    (&key(rleg-trans-pos #f(0 0 0))
	 (lleg-trans-pos #f(0 0 0))
	 (rleg-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad 0)))
	 (lleg-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad 0))))
  (let* ((rleg-end-coords (send *robot* :rleg :end-coords :copy-worldcoords))
	 (lleg-end-coords (send *robot* :lleg :end-coords :copy-worldcoords))
	 (rleg-translation-coords (send rleg-end-coords :translate rleg-trans-pos :world) :copy-worldcoords)
	 (lleg-translation-coords (send lleg-end-coords :translate lleg-trans-pos :world) :copy-worldcoords)
	 (rleg-target (make-coords :pos (send rleg-translation-coords :pos)
				   :rpy rleg-rpy))
	 (lleg-target (make-coords :pos (send lleg-translation-coords :pos)
				   :rpy lleg-rpy)))
    (send *robot* :fullbody-inverse-kinematics (list rleg-target lleg-target)
	  :move-target (list (send *robot* :rleg :end-coords)
			     (send *robot* :lleg :end-coords))
	  :link-list (list (send *robot* :link-list (send *robot* :rleg :end-coords :parent))
			   (send *robot* :link-list (send *robot* :lleg :end-coords :parent)))
	  :translation-axis (list t t)
	  :rotation-axis (list t t)
	  :target-centroid-pos (midpoint 0.8
	  				 (send *robot* :rleg :end-coords :worldpos)
	  				 (send *robot* :lleg :end-coords :worldpos))
	  ;; :cog-translation-axis :z
	  )
    )
  )

(defun reset-pose ()
  (send *robot* :reset-pose)
  (send *ri* :angle-vector (send *robot* :angle-vector))
  )

(defun init ()
    (send *robot* :angle-vector #f(0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 40.0 -20.0 -5.0 -80.0 0.0 0.0 -20.0 0.0 40.0 20.0 5.0 -80.0 0.0 0.0 -20.0 0.0 90.0 0.0 0.0 0.0 0.0 0.0 90.0 0.0 0.0 0.0 0.0))
  (send *ri* :angle-vector (send *robot* :angle-vector))
  (send *ri* :wait-interpolation)
  )

;; (defun lleg-back ()
;;   (send *robot* :angle-vector #f(0.0 0.0 -20.0 20.0 0.0 0.0 0.0 0.0 -20.0 20.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 40.0 -20.0 -5.0 -80.0 0.0 0.0 -20.0 0.0 40.0 20.0 5.0 -80.0 0.0 0.0 -20.0 0.0 90.0 0.0 0.0 0.0 0.0 0.0 90.0 0.0 0.0 0.0 0.0))
;;   (send *ri* :angle-vector (send *robot* :angle-vector))
;;   (send *ri* :wait-interpolation)
;;   (foot-solve-ik :lleg-trans-pos #f(-300 100 0) :lleg-rpy (float-vector (deg2rad 30) 0 0))
;;   (send *ri* :angle-vector (send *robot* :angle-vector))
;;   (send *ri* :wait-interpolation)
;;   )

(defun zmp-start ()
  (send *ri* :start-auto-balancer)
  (send *ri* :set-auto-stabilizer-param :default-zmp-offsets '(#f(0.05 0.02) #f(0.05 -0.02)))
  )

(defun zmp-stop ()
  (send *ri* :stop-auto-balancer)
  )

(defun lower-waist
  (&optional (dz 60) (tm 5000))
  ;(send *robot* :reset-pose)
  (send *robot* :fix-leg-to-coords (make-coords))
  (send *robot* :legs :move-end-pos (float-vector 0 0 dz))
  (send *robot* :move-centroid-on-foot :both '(:rleg :lleg))
  (send *ri* :angle-vector (send *robot* :angle-vector) tm)
  (send *ri* :wait-interpolation)
  )

(defun posture ()
    ;; (send *robot* :torso :waist-p :joint-angle 30)
    ;; (send *ri* :angle-vector (send *robot* :angle-vector))
    ;; (send *ri* :wait-interpolation)
    
  ;; (arm-solve-ik :larm-trans-pos #f(200 50 0) :larm-rpy (float-vector (deg2rad -20) 0 0) :rarm-trans-pos #f(200 -50 0) :rarm-rpy (float-vector (deg2rad 20) 0 0))
  ;; (send *ri* :angle-vector (send *robot* :angle-vector))
  ;; (send *ri* :wait-interpolation)

  (arm-solve-ik :larm-trans-pos #f(400 0 0) :rarm-trans-pos #f(400 0 0))
  (send *ri* :angle-vector (send *robot* :angle-vector))
  (send *ri* :wait-interpolation)
  
  (arm-solve-ik :larm-trans-pos #f(0 -100 100) :rarm-trans-pos #f(0 100 100))
  (send *ri* :angle-vector (send *robot* :angle-vector))
  (send *ri* :wait-interpolation)
  )

(defun hand ()
  (send *robot* :larm :wrist-p :joint-angle -60)
  (send *robot* :rarm :wrist-p :joint-angle -60)
  (send *robot* :rarm :wrist-r :joint-angle 50)
  (send *robot* :larm :wrist-r :joint-angle -50)
  (send *ri* :angle-vector (send *robot* :angle-vector))
  (send *ri* :wait-interpolation)
  )

(defun lie-hand ()
  (send *robot* :rarm :wrist-r :joint-angle 90)
  (send *robot* :larm :wrist-y :joint-angle 0)
  ;;(send *robot* :larm :wrist-r :joint-angle 90)
  (send *ri* :angle-vector (send *robot* :angle-vector))
  (send *ri* :wait-interpolation)
  )
;; (defun foot-posture ()
;;   (foot-solve-ik :lleg-trans-pos #f(-300 100 0) :lleg-rpy (float-vector (deg2rad 30) 0 0))
;;   (send *ri* :angle-vector (send *robot* :angle-vector))
;;   (send *ri* :wait-interpolation)
;;   )


(defun lleg-foot ()
  (send *ri* :set-foot-steps
	(list (make-coords :coords (send *robot* :lleg :end-coords :copy-worldcoords) :name :lleg)
	      (make-coords :coords (send
				    (send *robot* :rleg :end-coords :copy-worldcoords)
				    :translate (float-vector 0 -100 0))
			   :name :rleg)
	      (make-coords :coords (send
				    (send
				     (send *robot* :lleg :end-coords :copy-worldcoords)
				     :translate (float-vector -100 200 0))
				    :rotate (deg2rad 60) :z)
			   :name :lleg)
	       (make-coords :coords (send
				    (send *robot* :rleg :end-coords :copy-worldcoords)
				    :translate (float-vector 200 0 0))
			   :name :rleg)
	      )
	)
  )
	      
  ;send *robot* :torso :waist-y :joint-angle 30
(defun set-foot-steps ()
  (dotimes (i 6)
    (send *ri* :set-foot-steps
  	(list (make-coords :coords (send *robot* :rleg :end-coords :copy-worldcoords) :name :rleg)
  	      (make-coords :coords (send (send (send *robot* :lleg :end-coords :copy-worldcoords) :translate (float-vector -50 50 0)) :rotate (deg2rad 15) :z) :name :lleg)
  	      (make-coords :coords (send (send (send *robot* :rleg :end-coords :copy-worldcoords) :translate (float-vector -50 50 0)) :rotate (deg2rad 15) :z) :name :rleg)
  	      )
  	)
    )
)
;; (defun set-foot ()
;;   (send *ri* :set-foot-steps-with-param
;;         :foot-step-list
;;         (list (make-coords :coords (send *robot* :rleg :end-coords :copy-worldcoords) :name :rleg)
;;   	    (make-coords :coords (send (send (send *robot* :lleg :end-coords :copy-worldcoords) :translate (float-vector -100 100 0)) :rotate (deg2rad 30) :z) :name :lleg)
;;   	    (make-coords :coords (send (send (send *robot* :rleg :end-coords :copy-worldcoords) :translate (float-vector -100 100 0)) :rotate (deg2rad 30) :z) :name :rleg)
;;   	    )
;;         :step-time-list (list 2000 2000 2000)
;;         )
;; )
  ;; (send *ri* :set-foot-steps-with-base-height
  ;;       :fs
  ;;       (list (make-coords :coords (send *robot* :rleg :end-coords :copy-worldcoords) :name :rleg)
  ;; 	    (make-coords :coords (send (send (send *robot* :lleg :end-coords :copy-worldcoords) :translate (float-vector -100 100 0)) :rotate (deg2rad 30) :z) :name :lleg)
  ;; 	    (make-coords :coords (send (send (send *robot* :rleg :end-coords :copy-worldcoords) :translate (float-vector -100 100 0)) :rotate (deg2rad 30) :z) :name :rleg)
  ;; 	    )
  ;;       :av-list (list (send *robot* :angle-vector) (send *robot* :angle-vector) (send *robot* :angle-vector))
  ;;       :time-list (list 2000 2000 2000)
  ;;        )
  )


(defun bring ()
  ;; インピーダンス制御の設定
  (send *ri* :remove-force-sensor-offset-rmfo-arms) ;;手先力センサのオフセットを除去。
  (send *ri* :set-impedance-controller-param :arms :K-r 200.0 :D-r 300.0  :M-r 1
	:K-p 500.0 :D-p 1000.0 :M-p 5) ;;インピーダンス制御の設定。K(バネ)D(ダンパ)M(慣性)、　r(回転)、p(並進)
  (send *ri* :start-impedance :arms) ;;インピーダンス制御開始


  (send *ri* :set-ref-forces
        (list
         (float-vector 0 0 0) ;;rleg
         (float-vector 0 0 0) ;;lleg
         (float-vector 0 -100  0) ;;rarm (x, y, z) 単位はニュートン
         (float-vector 0 100   0) ;;larm
         )
         3000)
  (unix:sleep 5) ;;指定した時間(秒)処理を止める
  
  (send *ri* :set-ref-forces
        (list
         (float-vector 0 0 0)
         (float-vector 0 0 0)
         (float-vector 0 -100 -50) ;;rarm
         (float-vector 0 100  -50) ;;larm
         )
         5000)
  (unix:sleep 5)

  (send *robot* :arms :move-end-pos #f(0 0 100))
  (send *ri* :angle-vector (send *robot* :angle-vector) 5000)
  (send *ri* :wait-interpolation)
)

(defun rotate ()
  (send *ri* :set-foot-steps
	(list (make-coords :coords (send *robot* :lleg :end-coords :copy-worldcoords) :name :lleg)
	      (make-coords :coords (send
				    (send
				     (send *robot* :rleg :end-coords :copy-worldcoords)
				     :translate (float-vector 0 -150 0))
				    :rotate (deg2rad 90) :z)
			   :name :rleg)
	      )
	)
)

(defun down ()
  (send *robot* :arms :move-end-pos #f(0 0 -100))
  (send *ri* :angle-vector (send *robot* :angle-vector))
  (send *ri* :wait-interpolation)
  (send *ri* :stop-impedance :arms)
)

(defun all ()
  (lleg-foot)
  (lower-waist)
  (zmp-start)
  (posture)
  (bring)
  (rotate)
  (down)
)
