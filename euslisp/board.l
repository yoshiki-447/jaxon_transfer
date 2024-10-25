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

(defun posture (&key(run t))
  ;; (arm-solve-ik :larm-trans-pos #f(200 0 0) :rarm-trans-pos #f(200 0 0))
  ;; (send *ri* :angle-vector (send *robot* :angle-vector))
  ;; (send *ri* :wait-interpolation)
  ;; (arm-solve-ik :larm-trans-pos #f(0 -100 100) :rarm-trans-pos #f(0 100 -100))
  ;; (send *ri* :angle-vector (send *robot* :angle-vector))
  ;; (send *ri* :wait-interpolation)

  ;; (arm-solve-ik :larm-trans-pos #f(300 -200 250) :rarm-trans-pos #f(400 -50 0)
  ;; 		:rarm-rpy (float-vector (deg2rad -25) (deg2rad 0) (deg2rad 0))
  ;; 		:larm-rpy (float-vector (deg2rad -35) (deg2rad 0) (deg2rad 0)))
  
  ;;(send *robot* :reset-pose)
  (send *robot* :torso :waist-y :joint-angle -25)
  (send *robot* :torso :waist-p :joint-angle 25)
  ;; (arm-solve-ik-locate :larm-trans-pos #f(600 200 1000) :rarm-trans-pos #f(550 -300 700)
  ;; 		:rarm-rpy (float-vector (deg2rad -15) (deg2rad 0) (deg2rad 0))
  ;; 		:larm-rpy (float-vector (deg2rad -35) (deg2rad 0) (deg2rad 0)))
  (full-solve-ik-locate :larm-trans-pos #f(750 -300 1100) :rarm-trans-pos #f(400 -650 725)
  			:rarm-rpy (float-vector (deg2rad -35) (deg2rad 30) (deg2rad 0))
  			:larm-rpy (float-vector (deg2rad -35) (deg2rad 30) (deg2rad 0))
  			:centroid 0.2)
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

(defun generate-circle-trajectory (&key (start-pos) (angle) (num-points) (height) (larm nil))
  ;; centerは円の中心座標、radiusは半径、num-pointsは軌道上の点の数
  (let* (;; (center2hand (float-vector (aref (send *robot* :rarm :end-coords :worldpos) 0) (aref (send *robot* :rarm :end-coords :worldpos) 1) 0.0))
	 ;; (center (v- start-pos center2hand))
	 (center (float-vector (aref (send *robot* :chest_link0_lk :worldpos) 0) (aref (send *robot* :chest_link0_lk :worldpos) 0) (aref (send *robot* :rarm :end-coords :worldpos) 2)))
	 (center2hand (v- (send *robot* :rarm :end-coords :worldpos) center))
	 (radius (norm center2hand))
	 ;;(theta-step (/ (- (/ pi 2) angle) num-points))
	 (theta-step (/ (- (deg2rad 60) angle) num-points))  ;;60or90
	 (diff-z (/ height num-points))
	 ;; (ini-theta (atan (aref (send *robot* :rarm :end-coords :worldpos) 0) (aref (send *robot* :rarm :end-coords :worldpos) 1)))
	 (ini-theta (atan (aref center2hand 0) (aref center2hand 1)))
	 (trajectory '())) ;; 結果を蓄積するリスト
    (when larm
      ;; (setq center2hand (float-vector (aref (send *robot* :larm :end-coords :worldpos) 0) (aref (send *robot* :larm :end-coords :worldpos) 1) 0.0))
      ;; (setq center (v- start-pos center2hand))
      (setq center (float-vector (aref (send *robot* :chest_link0_lk :worldpos) 0) (aref (send *robot* :chest_link0_lk :worldpos) 0) (aref (send *robot* :larm :end-coords :worldpos) 2)))
      (setq center2hand (v- (send *robot* :larm :end-coords :worldpos) center))
      (setq radius (norm center2hand))
      (setq diff-z (/ height num-points))
      ;; (setq ini-theta (atan (aref (send *robot* :larm :end-coords :worldpos) 0) (aref (send *robot* :larm :end-coords :worldpos) 1)))
      (setq ini-theta (atan (aref center2hand 0) (aref center2hand 1)))
      ;;(setq theta-step (/ (- (/ pi 2) angle) num-points))
      )
    ;; (setq c1 (make-cube 100 100 100))
    ;; (setq c2 (make-cube 100 100 100))
    ;; (send c1 :locate center :world)
    (dotimes (i (+ num-points 1))
      ;; (print "Enter to next")(read-line)
      (let ((theta (* i theta-step)) (diff-z (* i diff-z)))
	(setq theta (- ini-theta theta))
	;; (setq theta (+ theta angle))
	(cond (larm
	       (setq target-circle (make-coords :pos (float-vector (* 1.0 (* radius (sin theta))) (* 1.0 (* radius (cos theta))) (- 0.0 diff-z))))
	       (setq circle (v+ center (send target-circle :pos)))
	       ;;(setq circle (v+ (send center :pos) (send target-circle :pos)))  ;;larm
	       )
	      (t
	       (setq target-circle (make-coords :pos (float-vector (* 1.0 (* radius (sin theta))) (* 1.0 (* radius (cos theta))) (- 0.0 diff-z))))
	       (setq circle (v+ center (send target-circle :pos)))
	       ;;(setq circle (v+ (send center :pos) (send target-circle :pos)));;rarm
	       )
	      )
        (cond ((> (length trajectory) 0)
	       ;; (send c2 :locate circle :world)
	       (nconc trajectory (list circle))
	       )
	      (t
	       ;; (send c2 :locate circle :world)
	       (push circle trajectory)
	       )
	      )
	)
      
      ;; (objects (list *robot* c1 c2))
      )
    trajectory
    )
  )

(defun follow-circle (&key (trajectory nil) (l_tra nil) (run nil) &allow-other-keys)
  (setq start-value 0.2)
  (setq finish-value 0.8)
  (setq diff (/ (- finish-value start-value) (- (length trajectory) 1)))
  (setq r-yaw -35)
  (setq l-yaw -35)
  ;; (setq hand-diff (/ 90 (- (length trajectory) 1)))
  (setq hand-diff (/ 70 (- (length trajectory) 1)))
  (dotimes (i (length trajectory))
    ;;(arm-solve-ik-locate :rarm-trans-pos target :rarm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad r-angle)))
    (setq r_target (elt trajectory i))
    (setq l_target (elt l_tra i))
    (full-solve-ik-locate :rarm-trans-pos r_target
    			  :larm-trans-pos l_target
    			  :rarm-rpy (float-vector (deg2rad r-yaw) (deg2rad 0) (deg2rad 0))
    			  ;;:rarm-rpy (float-vector (deg2rad 60) (deg2rad r-pitch) (deg2rad r-yaw)) ;;pattern2
    			  ;; :rarm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad (- l-start 90)))  ;;pattern1
    			  :larm-rpy (float-vector (deg2rad l-yaw) (deg2rad 0) (deg2rad 0))
    			  :centroid start-value)
    ;;(objects (list *robot* r_target))
    (print "Enter to next")(read-line)
    (unix:usleep 200000)
    (when run
      (send *ri* :angle-vector (send *robot* :angle-vector) 500)
      (send *ri* :wait-interpolation))
    (setq start-value (+ start-value diff))
    (setq r-yaw (+ r-yaw hand-diff))
    (setq l-yaw (+ l-yaw hand-diff))
    )
  )


(defun rotate (&rest args)
  ;; (apply #'follow-circle (append args (list :trajectory (generate-circle-trajectory :start-pos (send *robot* :rarm :end-coords :worldpos) :radius (norm (float-vector (aref (send *robot* :rarm :end-coords :worldpos) 0) (aref (send *robot* :rarm :end-coords :worldpos) 1))) :angle (deg2rad 0) :num-points 10 :height 50) :l_tra (generate-circle-trajectory :start-pos (send *robot* :larm :end-coords :worldpos) :radius (norm (float-vector (aref (send *robot* :larm :end-coords :worldpos) 0) (aref (send *robot* :larm :end-coords :worldpos) 1))) :angle (deg2rad 0) :num-points 10 :height 50 :larm t))))  ;;larm200or275
  (apply #'follow-circle (append args (list :trajectory (generate-circle-trajectory :start-pos (send *robot* :rarm :end-coords :worldpos) :angle (deg2rad 0) :num-points 10 :height 50) :l_tra (generate-circle-trajectory :start-pos (send *robot* :larm :end-coords :worldpos) :angle (deg2rad 0) :num-points 10 :height 50 :larm t))))
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
				       )
	  )
    (objects (list *robot* rarm-target))
    ;; (send *robot* :newcoords (make-coords))
    )
  )
