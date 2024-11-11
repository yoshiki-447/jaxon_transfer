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

(defun arm-solve-ik-locate
    (&key(rarm-trans-pos (send *robot* :rarm :end-coords :worldpos))
	 (larm-trans-pos (send *robot* :larm :end-coords :worldpos))
	 (rarm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad 0)))
	 (larm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad 0))))
  (let* ((rarm-end-coords (send *robot* :rarm :end-coords :copy-worldcoords))
	 (larm-end-coords (send *robot* :larm :end-coords :copy-worldcoords))
	 (rarm-translation-coords (send rarm-end-coords :locate rarm-trans-pos :world) :copy-worldcoords)
	 (larm-translation-coords (send larm-end-coords :locate larm-trans-pos :world) :copy-worldcoords)
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

(defun zmp-start ()
  (send *ri* :start-auto-balancer)
  (send *ri* :set-auto-stabilizer-param :default-zmp-offsets '(#f(0.05 0.02) #f(0.05 -0.02)))
  )

(defun zmp-stop ()
  (send *ri* :stop-auto-balancer)
  )

(defun move-waist (&key(dz) (tm) (run t))
       (send *robot* :fix-leg-to-coords (make-coords))
       (send *robot* :legs :move-end-pos (float-vector 0 0 dz))
       (send *robot* :move-centroid-on-foot :both '(:rleg :lleg))
       (when run
	 (send *ri* :angle-vector (send *robot* :angle-vector) tm)
	 (send *ri* :wait-interpolation)
	 )
       ;; (send *robot* :newcoords (make-coords))
       (send *robot* :fix-leg-to-coords (make-coords))
       )

(defun upper-waist
    (&key (dz -60) (tm 5000) (run t))
  (move-waist :dz dz :tm tm :run run)
  )

(defun lower-waist
    (&key (dz 60) (tm 5000) (run t))
  (move-waist :dz dz :tm tm :run run) 
  )

(defun lleg-foot (&key (run t))
  (let ((target-lleg-coords1 (send
			      (send
			       (send *robot* :lleg :end-coords :copy-worldcoords)
			       :translate (float-vector 0 100 0))
			      :rotate (deg2rad 25) :z)
			     )
	(target-lleg-coords2 (send
			      (send
			       (send *robot* :lleg :end-coords :copy-worldcoords)
			       :translate (float-vector 0 200 0))
			      :rotate (deg2rad 45) :z)
			     )
	(target-rleg-coords1 (send
			      (send
			       (send *robot* :rleg :end-coords :copy-worldcoords)
			       :translate (float-vector 0 -100 0))
			      :rotate (deg2rad -25) :z)
			      )
	(target-rleg-coords2 (send
			      (send
			       (send *robot* :rleg :end-coords :copy-worldcoords)
			       :translate (float-vector 0 -200 0))
			      :rotate (deg2rad -45) :z)
			     )
	;; (target-lleg-coords1 (send
	;; 		      (send
	;; 		       (send *robot* :lleg :end-coords :copy-worldcoords)
	;; 		       :translate (float-vector 0 200 0))
	;; 		      :rotate (deg2rad 30) :z)
	;; 		     )
	;; (target-lleg-coords2 (send
	;; 		      (send
	;; 		       (send *robot* :lleg :end-coords :copy-worldcoords)
	;; 		       :translate (float-vector -100 200 0))
	;; 		      :rotate (deg2rad 30) :z)
	;; 		     )
	)
    (cond (run
	   (send *ri* :set-foot-steps
		 (list (make-coords :coords (send *robot* :rleg :end-coords :copy-worldcoords) :name :rleg)
		       (make-coords :coords target-lleg-coords1 :name :lleg)
		       ;;(make-coords :coords (send *robot* :rleg :end-coords :copy-worldcoords) :name :rleg)
		       (make-coords :coords target-rleg-coords1 :name :rleg)
		       (make-coords :coords target-lleg-coords2 :name :lleg)
		       (make-coords :coords target-rleg-coords2 :name :rleg)
		       )
		 )
	       (send *robot* :angle-vector (send *ri* :state :potentio-vector))
	       )
	  (t (send *robot* :lleg :inverse-kinematics target-lleg-coords1)
	     (send *robot* :lleg :inverse-kinematics target-lleg-coords2)
	     (send *robot* :rleg :inverse-kinematics target-rleg-coords1)
	     (send *robot* :rleg :inverse-kinematics target-rleg-coords2)
	     )
	  )

    (send *robot* :fix-leg-to-coords (make-coords))
    (send *robot* :move-centroid-on-foot :both '(:rleg :lleg))
    )
  )

(defun reset-pose (&key(run t))
  (send *robot* :reset-pose)
  (when run
    (send *ri* :angle-vector (send *robot* :angle-vector) 5000)
    )
  (send *robot* :fix-leg-to-coords (make-coords))
  )

(defun torso-45 (&key(run t))
  (send *robot* :torso :waist-y :joint-angle -45)
  (when run
    (send *ri* :angle-vector (send *robot* :angle-vector) 5000)
    (send *ri* :wait-interpolation)
    )
  )

(defun torso20 (&key(run t))
  (send *robot* :torso :waist-y :joint-angle 20)
  (when run
    (send *ri* :angle-vector (send *robot* :angle-vector) 10000)
    (send *ri* :wait-interpolation)
    )
  )

(defun sim-irt ()
  (send *robot* :angle-vector (send *ri* :state :potentio-vector))
  (send *robot* :fix-leg-to-coords (make-coords))
  )

(defun irt-ri ()
  (send *ri* :angle-vector (send *robot* :angle-vector) 10000)
  (send *ri* :wait-interpolation)
  )

(defun posture (&key(run t))
  ;;(send *robot* :reset-pose)
  (send *robot* :torso :waist-y :joint-angle -25)
  (send *robot* :torso :waist-p :joint-angle 25)
  (full-solve-ik-locate :larm-trans-pos #f(700 -350 1000) :rarm-trans-pos #f(400 -650 725)
  			:rarm-rpy (float-vector (deg2rad -35) (deg2rad 30) (deg2rad 0))
  			:larm-rpy (float-vector (deg2rad -35) (deg2rad 30) (deg2rad 0))
  			:centroid 0.2)
  ;; (full-solve-ik-locate :larm-trans-pos #f(750 -300 1100) :rarm-trans-pos #f(400 -650 725)
  ;; 			:rarm-rpy (float-vector (deg2rad -35) (deg2rad 30) (deg2rad 0))
  ;; 			:larm-rpy (float-vector (deg2rad -35) (deg2rad 30) (deg2rad 0))
  ;; 			:centroid 0.2)  ;;２回目の実験、肘が上に来ているので回転後の左手が変
  ;; (full-solve-ik-locate :larm-trans-pos #f(800 -300 1000) :rarm-trans-pos #f(400 -650 725)
  ;; 			:rarm-rpy (float-vector (deg2rad -35) (deg2rad 30) (deg2rad 0))
  ;; 			:larm-rpy (float-vector (deg2rad -35) (deg2rad 30) (deg2rad 0))
  ;; 			:centroid 0.2)  ;;一回目の実験、左手ちょい低い、軽マネキンだと左右の腕の幅長い
  ;; (full-solve-ik-locate :larm-trans-pos #f(750 -200 1100) :rarm-trans-pos #f(450 -550 750)
  ;; 			:rarm-rpy (float-vector (deg2rad -35) (deg2rad 0) (deg2rad 0))
  ;; 			:larm-rpy (float-vector (deg2rad -25) (deg2rad 0) (deg2rad 0))
  ;; 			:centroid 0.2) ;;link-listをchestにする前
  ;; (full-solve-ik-locate :larm-trans-pos #f(800 -250 1100) :rarm-trans-pos #f(500 -600 750)
  ;; 		:rarm-rpy (float-vector (deg2rad -35) (deg2rad 0) (deg2rad 0))
  ;; 		:larm-rpy (float-vector (deg2rad -25) (deg2rad 0) (deg2rad 0))
  ;; 		:centroid 0.2) ;;rarm長い
  ;; (full-solve-ik-locate :larm-trans-pos #f(750 -100 1100) :rarm-trans-pos #f(600 -500 750)
  ;; 		:rarm-rpy (float-vector (deg2rad -25) (deg2rad 0) (deg2rad 0))
  ;; 		:larm-rpy (float-vector (deg2rad -15) (deg2rad 0) (deg2rad 0))
  ;; 		:centroid 0.2)  ;;ちょっと傾いている
  ;; (full-solve-ik-locate :larm-trans-pos #f(600 -100 1050) :rarm-trans-pos #f(350 -500 700)
  ;; 		:rarm-rpy (float-vector (deg2rad -15) (deg2rad 0) (deg2rad 0))
  ;; 		:larm-rpy (float-vector (deg2rad -35) (deg2rad 0) (deg2rad 0))
  ;; 		:centroid 0.2)
  ;; (full-solve-ik-locate :larm-trans-pos #f(600 200 1000) :rarm-trans-pos #f(550 -300 700)
  ;; 		:rarm-rpy (float-vector (deg2rad -15) (deg2rad 0) (deg2rad 0))
  ;; 		:larm-rpy (float-vector (deg2rad -35) (deg2rad 0) (deg2rad 0))
  ;; 		:centroid 0.2)
  (send *robot* :rarm :wrist-p :joint-angle -40)
  (send *robot* :larm :wrist-p :joint-angle 0)
  (when run
    (send *ri* :angle-vector (send *robot* :angle-vector) 10000)
    (send *ri* :wait-interpolation)
    )
  )

(defun impedance ()
  (send *ri* :set-impedance-controller-param :arms :K-r 200.0 :D-r 300.0  :M-r 1
	:K-p 500.0 :D-p 1000.0 :M-p 5)
  (send *ri* :start-impedance :rarm)
  (send *ri* :set-ref-forces
	(list
	 (float-vector 0 0 0) ;;rleg
	 (float-vector 0 0 0) ;;lleg
	 (float-vector 0 0 0) ;;rarm (x, y, z) 単位はニュートン
	 (float-vector 0 0 0) ;;larm
	 )
	3000)
  )

(defun all ()
  (lleg-foot :run nil)
  (send *ri* :angle-vector (send *robot* :angle-vector) 10000)
  (lower-waist)
  (posture)
  (impedance)
  (rotate :run t)
  )

(defvar *wrench-list* '())
(defun generate-circle-trajectory (&key (r-start-pos) (r-angle) (r-height) (l-start-pos) (l-angle) (l-height) (num-points))
  (setq *wrench-list* '())
  (setq mass 20.0)
  (setq grav 9.8)
  (setq force_fit 100.0) ;;?
  (let* (;; (r-center2hand (float-vector (aref (send *robot* :rarm :end-coords :worldpos) 0) (aref (send *robot* :rarm :end-coords :worldpos) 1) 0.0))
	 ;; (r-center (v- r-start-pos r-center2hand))
	 (r-center (float-vector (aref (send *robot* :chest_link0_lk :worldpos) 0) (aref (send *robot* :chest_link0_lk :worldpos) 1) (aref (send *robot* :rarm :end-coords :worldpos) 2)))
	 (r-center2hand (v- (send *robot* :rarm :end-coords :worldpos) r-center))
	 (r-radius (norm r-center2hand))
	 (r-diff-z (/ r-height num-points))
	 ;; (r-ini-theta (atan (aref (send *robot* :rarm :end-coords :worldpos) 0) (aref (send *robot* :rarm :end-coords :worldpos) 1)))
	 (r-ini-theta (atan (aref r-center2hand 0) (aref r-center2hand 1)))
	 
	 ;; (l-center2hand (float-vector (aref (send *robot* :larm :end-coords :worldpos) 0) (aref (send *robot* :larm :end-coords :worldpos) 1) 0.0))
	 ;; (l-center (v- l-start-pos l-center2hand))
	 ;; (l-center (float-vector 0.0 0.0 (aref (send *robot* :larm :end-coords :worldpos) 2)))
	 (l-center (float-vector (aref (send *robot* :chest_link0_lk :worldpos) 0) (aref (send *robot* :chest_link0_lk :worldpos) 1) (aref (send *robot* :larm :end-coords :worldpos) 2)))
	 (l-center2hand (v- (send *robot* :larm :end-coords :worldpos) l-center))
	 (l-radius (norm l-center2hand))
	 (l-diff-z (/ l-height num-points))
	 ;; (setq ini-theta (atan (aref (send *robot* :larm :end-coords :worldpos) 0) (aref (send *robot* :larm :end-coords :worldpos) 1)))
	 (l-ini-theta (atan (aref l-center2hand 0) (aref l-center2hand 1)))
	 ;;(setq theta-step (/ (- (/ pi 2) angle) num-points))
	 (r-theta-step (/ (- (deg2rad 60) r-angle) num-points))  ;;60or90
	 (l-theta-step (/ (- (deg2rad 60) l-angle) num-points))  ;;60or90
	 (r-trajectory '())
	 (l-trajectory '()))
    ;; (setq c1 (make-cube 100 100 100))
    ;; (setq c2 (make-cube 100 100 100))
    ;; (send c1 :locate center :world)
    (dotimes (i (+ num-points 1))
      ;; (print "Enter to next")(read-line)
      (let ((r-theta (* i r-theta-step)) (l-theta (* i l-theta-step)) (r-diff-z (* i r-diff-z)) (l-diff-z (* i l-diff-z)))
	(setq r-theta (- r-ini-theta r-theta))
	(setq l-theta (- l-ini-theta l-theta))
	;; (setq theta (+ theta angle))

	(setq r-target-circle (float-vector (* 1.0 (* r-radius (sin r-theta))) (* 1.0 (* r-radius (cos r-theta))) (- 0.0 r-diff-z)))
	(setq r-circle (v+ r-center r-target-circle))
	;;(setq r-circle (v+ (send r-center :pos) (send r-target-circle :pos)));;rarm
	
	(setq l-target-circle (float-vector (* 1.0 (* l-radius (sin l-theta))) (* 1.0 (* l-radius (cos l-theta))) (- 0.0 l-diff-z)))
	(setq l-circle (v+ l-center l-target-circle))
	;;(setq l-circle (v+ (send l-center :pos) (send l-target-circle :pos)))  ;;larm

	(setq r_force (/ (* (* mass grav) (cos r-theta)) 2.0))
	(setq r_moment (* r_force r-radius))
	(setq r_fx 0.0)
	(setq r_fy (+ (* r_force (sin r-theta)) (* force_fit (cos r-theta))))
	(setq r_fz (- (* r_force (cos r-theta)) (* force_fit (sin r-theta))))
	(setq r_Mx 0)
	(setq r_My (* r_moment (sin r-theta)))
	(setq r_Mz (* r_moment (cos r-theta)))
	(setq r_wr (list r_fx r_fy r_fz r_Mx r_My r_Mz))
	(setq l_force 50.0)
	(setq l_fx 0.0)
	(setq l_fy (- 0.0 (* l_force (sin l-theta))))
	(setq l_fz (- 0.0 (* l_force (cos l-theta))))
	(setq l_wr (list l_fx l_fy l_fz 0.0 0.0 0.0))
	(setq rl_wr (list 0.0 0.0 0.0 0.0 0.0 0.0))
	(setq ll_wr (list 0.0 0.0 0.0 0.0 0.0 0.0))
	(setq wr (append r_wr l_wr rl_wr ll_wr))
	(cond ((> (length *wrench-list*) 0)
	       (nconc *wrench-list* (list wr))
	       )
	      (t
	       (push wr *wrench-list*)
	       )
	      )
	)   
      (cond ((> (length r-trajectory) 0)
	     ;; (send c2 :locate circle :world)
	     (nconc r-trajectory (list r-circle))
	     )
	    (t
	     ;; (send c2 :locate circle :world)
	     (push r-circle r-trajectory)
	     )
	      )
      (cond ((> (length l-trajectory) 0)
	     ;; (send c2 :locate circle :world)
	     (nconc l-trajectory (list l-circle))
	     )
	    (t
	     ;; (send c2 :locate circle :world)
	     (push l-circle l-trajectory)
	     )
	    )
      )
    (setq trajectory '())
    (push l-trajectory trajectory) ;; elt trajectory 1
    (push r-trajectory trajectory) ;; elt trajectory 0
    trajectory
    )
  )

(defvar *angle-list* '())
(defvar *time-list* '())
(defun follow-circle (&key (r_tra nil) (l_tra nil) (run nil) &allow-other-keys)
  (setq start-value 0.2)
  (setq finish-value 0.8)
  (setq diff (/ (- finish-value start-value) (- (length r_tra) 1)))
  (setq r-yaw -35.0)
  (setq l-yaw -35.0)
  (setq *angle-list* '())
  (setq *time-list* '())
  (send *ri* :set-interpolation-mode :linear)
  ;; (setq hand-diff (/ 90 (- (length r_tra) 1)))
  (setq hand-diff (/ 70.0 (- (length r_tra) 1)))  ;;70
  (setq l-hand-diff (/ 20.0 (- (length l_tra) 1)))
  (dotimes (i (length r_tra))
    ;;(arm-solve-ik-locate :rarm-trans-pos target :rarm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad r-angle)))
    (setq r_target (elt r_tra i))
    (setq l_target (elt l_tra i))
    ;; (progn (reset-pose :run nil) (lleg-foot :run nil) (lower-waist :run nil) (posture :run nil))
    (full-solve-ik-locate :rarm-trans-pos r_target
    			  :larm-trans-pos l_target
    			  :rarm-rpy (float-vector (deg2rad r-yaw) (deg2rad 0) (deg2rad 0))
    			  ;;:rarm-rpy (float-vector (deg2rad 60) (deg2rad r-pitch) (deg2rad r-yaw)) ;;pattern2
    			  ;; :rarm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad (- l-start 90)))  ;;pattern1
    			  :larm-rpy (float-vector (deg2rad l-yaw) (deg2rad 0) (deg2rad 0))
    			  :centroid start-value)
    ;;(objects (list *robot* r_target))
    ;;(print "Enter to next")(read-line)
    ;;(unix:usleep 50000)
    ;; (when run
    ;;   (send *ri* :angle-vector (send *robot* :angle-vector) 500)
    ;;   (send *ri* :wait-interpolation))
    (setq start-value (+ start-value diff))
    (setq r-yaw (+ r-yaw hand-diff))
    (setq l-yaw (+ l-yaw l-hand-diff))
    
    (setq position (send *robot* :angle-vector))
    (cond ((> (length *angle-list*) 0)
  	   (nconc *angle-list* (list position))
  	   )
  	  (t
  	   (push position *angle-list*)
  	   )
  	  )
    (push 2.0 *time-list*)
    )
  ;;(format t "angle-list: ~A~%" *angle-list*)
  ;;(format t "time-list: ~A~%" *time-list*)
  ;;angle-list
  ;;time-list
  (when run
    (send *ri* :angle-vector (car *angle-list*))
    (send *ri* :wait-interpolation)
    (send *ri* :angle-vector-sequence-full *angle-list* *time-list*)
    (send *ri* :wait-interpolation)
    )
  )


(defun rotate (&rest args)
  (setq tra (generate-circle-trajectory :r-start-pos (send *robot* :rarm :end-coords :worldpos) :r-angle (deg2rad 0) :r-height 50.0 :l-start-pos (send *robot* :larm :end-coords :worldpos) :l-angle (deg2rad 0) :l-height 50.0 :num-points 5000))
  (apply #'follow-circle (append args (list :r_tra (elt tra 0) :l_tra (elt tra 1))))
  )

(defun impedance ()
  (send *ri* :set-impedance-controller-param :arms :K-r 200.0 :D-r 300.0  :M-r 1
	:K-p 500.0 :D-p 1000.0 :M-p 5)  ;;default :K-p 500.0 :D-p 1000.0 :M-p 5
  (send *ri* :start-impedance :arms)
  ;; (send *ri* :set-ref-forces
  ;; 	(list
  ;; 	 (float-vector 0 0 0) ;;rleg
  ;; 	 (float-vector 0 0 0) ;;lleg
  ;; 	 (float-vector 0 0 0) ;;rarm (x, y, z) 単位はニュートン
  ;; 	 (float-vector 0 0 0) ;;larm
  ;; 	 )
  ;; 	3000)
  )

(defun angle-drawplot (&key (list *angle-list*) (num 14)) ;;torso-y 14 ;;rarm-wrist-p 24 ;;rarm-wrist-r 23 ;;rarm-wrist-y 22
  (setq a (gnuplot))
  (setq b '())
  (dotimes (i (length list))
    (push (aref (elt list i) num) b)
    )
  (nreverse b)
  (send a :draw b)
  )

(defun wrench-drawplot (&key (list *wrench-list*) (num 1))
  (setq a (gnuplot))
  (setq b '())
  (dotimes (i (length list))
    (push (elt (elt list i) num) b)
    )
  (nreverse b)
  (send a :draw b)
  )

(defun full-solve-ik-locate
    (&key(rarm-trans-pos (send *robot* :rarm :end-coords :worldpos))
	 (larm-trans-pos (send *robot* :larm :end-coords :worldpos))
	 (rarm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad 0)))
	 (larm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad 0)))
	 (centroid 0.5))
  (let* ((rleg-coords (send *robot* :rleg :end-coords :copy-worldcoords))
	 (lleg-coords (send *robot* :lleg :end-coords :copy-worldcoords))
	 (rarm-end-coords (send *robot* :rarm :end-coords :copy-worldcoords))
	 (larm-end-coords (send *robot* :larm :end-coords :copy-worldcoords))
	 (rarm-translation-coords (send rarm-end-coords :locate rarm-trans-pos :world) :copy-worldcoords)
	 (larm-translation-coords (send larm-end-coords :locate larm-trans-pos :world) :copy-worldcoords)
	 (rarm-target (make-coords :pos (send rarm-translation-coords :pos)
				   :rpy rarm-rpy))
	 (larm-target (make-coords :pos (send larm-translation-coords :pos)
				   :rpy larm-rpy)))
    (send *robot* :fullbody-inverse-kinematics
	  (list rleg-coords lleg-coords rarm-target larm-target)
	  :move-target (list (send *robot* :rleg :end-coords)
			     (send *robot* :lleg :end-coords)
			     (send *robot* :rarm :end-coords)
			     (send *robot* :larm :end-coords))
	  :link-list (list (send *robot* :link-list (send *robot* :rleg :end-coords :parent))
			   (send *robot* :link-list (send *robot* :lleg :end-coords :parent))
			   ;; (send *robot* :link-list (send *robot* :rarm :end-coords :parent) (send *robot* :rarm :root-link))
			   (send *robot* :link-list (send *robot* :rarm :end-coords :parent) (send *robot* :chest_link2_lk))
			   ;; (send *robot* :link-list (send *robot* :rarm_link6_lk))
			   ;; (send *robot* :link-list (send *robot* :larm :end-coords :parent) (send *robot* :larm :root-link))
			   (send *robot* :link-list (send *robot* :larm :end-coords :parent) (send *robot* :chest_link2_lk))
			   ;; (send *robot* :link-list (send *robot* :larm_link6_lk))
			   )
	  :translation-axis (list t t t t) 
	  ;; :rotation-axis (list t t nil nil)
	  ;; :rotation-axis (list t t t nil)
	  ;; :rotation-axis (list t t t :z)
	  :rotation-axis (list t t t t)
	  :target-centroid-pos (midpoint centroid
	  				 (v+ (send *robot* :rleg :end-coords :worldpos) #f(0 0 0))
	  				 (send *robot* :lleg :end-coords :worldpos))
	  :cog-translation-axis :z
	  :root-link-virtual-joint-weight #f(0.1 0.1 0.1 0.1 1 1)
	  :addtional-weight-list (list (list (send *robot* :torso :waist-y :child-link) 1000.0)
	  			       (list (send *robot* :rleg :crotch-y :child-link) 100.0)
	  			       (list (send *robot* :lleg :crotch-y :child-link) 100.0)
	  			       (list (send *robot* :torso :waist-r :child-link) 0)
	  			       (list (send *robot* :torso :waist-p :child-link) 1000.0)
				       (list (send *robot* :larm :collar-y :child-link) 0.1)
				       (list (send *robot* :larm :wrist-r :child-link) 0.1)
				       )
	  )
    ;; (objects (list *robot* rarm-target))
    ;; (send *robot* :newcoords (make-coords))
    )
  )

;;progn (reset-pose :run nil) (lleg-foot :run nil) (lower-waist :run nil) (posture :run nil)
;;(send *ri* :angle-vector (send *robot* :angle-vector) 10000)
