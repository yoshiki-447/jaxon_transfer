(require "models/arrow-object.l")

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

(defun initial-pos ()
  ;; (arm-solve-ik :larm-trans-pos #f(300 0 200) :rarm-trans-pos #f(300 0 0)
  ;; 		:larm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad 0))
  ;; 		:rarm-rpy (float-vector (deg2rad 40) (deg2rad 0) (deg2rad 0)))
  ;; (send *ri* :angle-vector (send *robot* :angle-vector) 5000)
  ;; (send *ri* :wait-interpolation)
  (arm-solve-ik :larm-trans-pos #f(300 0 200)
		:larm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad 0)))
  (send *ri* :angle-vector (send *robot* :angle-vector) 5000)
  (send *ri* :wait-interpolation)
  (arm-solve-ik :larm-trans-pos #f(0 0 -200)
		:larm-rpy (float-vector (deg2rad 0) (deg2rad 60) (deg2rad 60)))
		;;:rarm-rpy (float-vector (deg2rad 40) (deg2rad 0) (deg2rad 0)))  ;;初期姿勢
  (send *ri* :angle-vector (send *robot* :angle-vector) 5000)
  (send *ri* :wait-interpolation)
  )

(defun bed-knee ()
  (arm-solve-ik :larm-trans-pos #f(-200 0 0)
		:larm-rpy (float-vector (deg2rad 0) (deg2rad 60) (deg2rad 60)))
		;;:rarm-rpy (float-vector (deg2rad 40) (deg2rad 0) (deg2rad 0)))  ;;被介助者の膝を出す
  (send *ri* :angle-vector (send *robot* :angle-vector) 5000)
  (send *ri* :wait-interpolation)
  )

(defun support ()
  (arm-solve-ik :larm-trans-pos #f(200 0 100)
		:larm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad 0)))
		;;:rarm-rpy (float-vector (deg2rad 40) (deg2rad 0) (deg2rad 0)))
  (send *ri* :angle-vector (send *robot* :angle-vector) 5000)  ;;左手で腰(or膝)を支える
  (send *ri* :wait-interpolation)
  )

(defun rarm-neck ()
  (full-solve-ik :rarm-trans-pos #f(300 0 200)
		 :rarm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad 0)) :centroid 0.2)
  (send *ri* :angle-vector (send *robot* :angle-vector) 5000)
  (send *ri* :wait-interpolation)
  (send *robot* :rarm :wrist-r :joint-angle 40)
  (send *ri* :angle-vector (send *robot* :angle-vector) 5000)
  (send *ri* :wait-interpolation)
  )

(defun finish-pos (&key (run t))
  (send *robot* :reset-pose)
  ;;(upper-waist)
  ;;(arm-solve-ik :larm-trans-pos #f(0 0 -150))
  (send *robot* :torso :waist-r :joint-angle -5)
  ;;(send *robot* :torso :waist-y :joint-angle 40)
  
  (send *robot* :rarm :collar-y :joint-angle 60)
  (send *robot* :rarm :shoulder-y :joint-angle -30)
  (send *robot* :rarm :shoulder-p :joint-angle 0)
  (send *robot* :rarm :shoulder-r :joint-angle -60.0)
  (send *robot* :rarm :elbow-p :joint-angle -80)
  (send *robot* :rarm :wrist-y :joint-angle 0)
  (send *robot* :rarm :wrist-p :joint-angle -20.0)
  (send *robot* :rarm :wrist-r :joint-angle 40)

  ;; (send *robot* :larm :shoulder-y :joint-angle -30)
  ;; (send *robot* :larm :shoulder-p :joint-angle -20)
  (send *robot* :larm :wrist-p :joint-angle 0)
  (send *robot* :larm :wrist-r :joint-angle 5.90432)
  (send *robot* :larm :wrist-y :joint-angle 14.7854)
  (send *robot* :larm :collar-y :joint-angle -20)
  (send *robot* :larm :shoulder-y :joint-angle -30)
  (send *robot* :larm :shoulder-p :joint-angle 12.713)
  (send *robot* :larm :shoulder-y :joint-angle 7.57822)
  (send *robot* :larm :shoulder-r :joint-angle 18.2888)
  (send *robot* :larm :elbow-p :joint-angle -88.4984)
  
  ;; (send *robot* :larm :shoulder-y :joint-angle -10)
  ;; (send *robot* :larm :shoulder-p :joint-angle 10)
  ;; (send *robot* :larm :shoulder-r :joint-angle 20)
  ;; (send *robot* :larm :elbow-p :joint-angle -50)
  ;; (send *robot* :larm :wrist-y :joint-angle 0)
  ;; (send *robot* :larm :wrist-p :joint-angle -20.0)
  ;; (send *robot* :larm :wrist-r :joint-angle 0.0)

  (when run
    (send *ri* :angle-vector (send *robot* :angle-vector) 10000)
    (send *ri* :wait-interpolation)
    )
  )

;; (defun posture ()
;;   (lleg-foot)
;;   (initial-pos)
;;   (bed-knee)
;;   (support)
;;   (lower-waist)
;;   (rarm-neck)
;;   (wake)
;;   )

(defun reset-pose (&key(run t))
  (send *robot* :reset-pose)
  (when run
    (send *ri* :change-cop 0.0 0.02 0.0 -0.02 5.0)
    (send *ri* :wait-interpolation)
    (send *ri* :angle-vector (send *robot* :angle-vector) 5000)
    (send *ri* :wait-interpolation)
    )
  (send *robot* :fix-leg-to-coords (make-coords))
  )

(defun sim-irt ()
  (send *robot* :angle-vector (send *ri* :state :potentio-vector))
  (send *robot* :fix-leg-to-coords (make-coords))
  )

(defun irt-ri ()
  (send *ri* :angle-vector (send *robot* :angle-vector) 10000)
  (send *ri* :wait-interpolation)
  )

(defun lleg-foot (&key (run t))
  (let ((target-lleg-coords (send (send *robot* :lleg :end-coords :copy-worldcoords)
				  :translate (float-vector 0 150 0))
			    ;; :rotate (deg2rad 60) :z)))
			    )
	(target-rleg-coords (send (send *robot* :rleg :end-coords :copy-worldcoords)
				  :translate (float-vector 0 -150 0))
			    ;; :rotate (deg2rad 60) :z)))
			    ))
    (cond (run
	   (send *ri* :set-foot-steps
		 (list (make-coords :coords (send *robot* :rleg :end-coords :copy-worldcoords) :name :rleg)
		       (make-coords :coords target-lleg-coords :name :lleg)
		       (make-coords :coords target-rleg-coords :name :rleg)
		       )
		 )
	       (send *robot* :angle-vector (send *ri* :state :potentio-vector))
	       )
	  (t
	   (send *robot* :lleg :inverse-kinematics target-lleg-coords)
	   (send *robot* :rleg :inverse-kinematics target-rleg-coords)
	   )
	  )

    (send *robot* :fix-leg-to-coords (make-coords))
    (send *robot* :move-centroid-on-foot :both '(:rleg :lleg))
    )
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
  ;;(send *robot* :reset-pose)
  ;; (send *robot* :fix-leg-to-coords (make-coords))
  ;; (send *robot* :legs :move-end-pos (float-vector 0 0 dz))
  ;; (send *robot* :move-centroid-on-foot :both '(:rleg :lleg))
  ;; (send *ri* :angle-vector (send *robot* :angle-vector) tm)
  ;; (send *ri* :wait-interpolation)
  ;; (send *robot* :newcoords (make-coords))
  )

(defun start-pos (&key (run t))
  ;; ;; (send *robot* :larm :shoulder-p :joint-angle 20)
  ;; (send *robot* :larm :shoulder-p :joint-angle -10)
  ;; (send *robot* :larm :collar-y :joint-angle -10)
  ;; (send *robot* :larm :shoulder-y :joint-angle 10)
  ;; (send *robot* :larm :elbow-p :joint-angle -70)

  ;; (send *robot* :torso :waist-r :joint-angle 5)
  ;; (send *robot* :torso :waist-p :joint-angle 20)

  ;; (send *robot* :rarm :shoulder-p :joint-angle -40)
  ;; (send *robot* :rarm :shoulder-r :joint-angle -30)
  ;; (send *robot* :rarm :elbow-p :joint-angle -30)

  ;; (send *robot* :rarm :collar-y :joint-angle 20)
  
  ;; ;; (arm-solve-ik :larm-trans-pos #f(0 0 100))
  ;; (send *robot* :larm :wrist-y :joint-angle -40)
  ;; (send *robot* :larm :wrist-r :joint-angle 40)
  
  ;; (send *robot* :rarm :wrist-r :joint-angle 60)
  ;; (send *robot* :rarm :wrist-p :joint-angle -60)

  ;; (send *robot* :fix-leg-to-coords (make-coords))
  ;; ;;(send *robot* :move-centroid-on-foot :both '(:rleg :lleg))

  ;; (when run
  ;;   (send *ri* :angle-vector (send *robot* :angle-vector) 5000)
  ;;   (send *ri* :wait-interpolation)
  ;;   )
  ;;pattern1
  (send *robot* :torso :waist-p :joint-angle 20)
  (full-solve-ik-locate :rarm-trans-pos #f(650 -400 950)
  			:larm-trans-pos #f(650 250 1200)
  			:rarm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad 90))
  			:larm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad 90))
  			:centroid 0.2) 
  ;; (send *robot* :rarm :middle-r :joint-angle 60)  
  ;; (send *robot* :rarm :index-r1 :joint-angle 60)
  ;;pattern2
  ;; (send *robot* :torso :waist-p :joint-angle 20)
  ;; (full-solve-ik-locate :rarm-trans-pos #f(650 -200 950)
  ;; 			:larm-trans-pos #f(500 250 1100) ;;#f(500 250 1200)
  ;; 			:rarm-rpy (float-vector (deg2rad 60) (deg2rad -30) (deg2rad 30))
  ;; 			:larm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad 90))
  ;; 			:centroid 0.2)
  
  ;;:larm-end-coords  595.177 379.498 1035.012 / -0.0 0.004 1.57>
  ;;:rarm-end-coords  682.921 -173.113 960.487 / 1.047 -0.26 0.264>
  (when run
    (send *ri* :angle-vector (send *robot* :angle-vector) 5000)
    (send *ri* :wait-interpolation)
    (send *ri* :change-cop 0.0 -0.1 0.0 -0.1 5.0)
    )
  )

(defun theta-cos (&key(a) (b) (index) (c))
  (setq theta (+ (* a (cos (* b index))) c))
  theta
  )

(defun theta-3 (&key (x1) (y1) (x2) (y2) (index))
  (setq mat (matrix (list (expt x1 3) (expt x1 2) x1 1)
		    (list (expt x2 3) (expt x2 2) x2 1)
		    (list (* 3 (expt x1 2)) (* 2 x1) 1 0)
		    (list (* 3 (expt x2 2)) (* 2 x2) 1 0)
		    ))
  (setq vec (float-vector y1 y2 0 0))
  (setq coeffs (transform (inverse-matrix mat) vec))
  (setq ax3 (* (aref coeffs 0) (expt index 3)))
  (setq bx2 (* (aref coeffs 1) (expt index 2)))
  (setq cx (* (aref coeffs 2) index))
  (setq d (aref coeffs 3))
  (setq y (+ ax3 bx2 cx d))
  y
  )

(defun vector-to-matrix (vec)
  (let ((vx (aref vec 0))
	(vy (aref vec 1))
	(vz (aref vec 2)))
    (let ((matrix (make-matrix 3 3)))
      (setf (aref matrix 0 0) 0
            (aref matrix 0 1) (- vz)
            (aref matrix 0 2) vy
            (aref matrix 1 0) vz
            (aref matrix 1 1) 0
            (aref matrix 1 2) (- vx)
            (aref matrix 2 0) (- vy)
            (aref matrix 2 1) vx
            (aref matrix 2 2) 0)
      matrix
      )
    )
  )

(defun transform-matrix (&key (rarm t))
  (cond (rarm
	 (setq R_ (send *robot* :rarm :end-coords :worldrot))
	 (setq p_ (send *robot* :rarm :end-coords :worldpos))
	 ;; (setq p_ (scale (- 1.0) p_))
	 )
	(t
	 (setq R_ (send *robot* :larm :end-coords :worldrot))
	 (setq p_ (send *robot* :larm :end-coords :worldpos))
	 ;; (setq p_ (scale (- 1.0) p_))
	 )
	)
  (setq p2R (m* (vector-to-matrix p_) R_))
  (setq m-zero (make-matrix 3 3))
  (let ((matrix (make-matrix 6 6)))
    (dotimes (i 3)
      (dotimes (j 3)
        (setf (aref matrix i j) (aref R_ i j)))) ;;左上
    (dotimes (i 3)
      (dotimes (j 3)
        (setf (aref matrix i (+ j 3)) (aref m-zero i j))))  ;;右上
    (dotimes (i 3)
      (dotimes (j 3)
        (setf (aref matrix (+ i 3) j) (aref p2R i j))))  ;; 左下
    (dotimes (i 3)
      (dotimes (j 3)
        (setf (aref matrix (+ i 3) (+ j 3)) (aref R_ i j))))  ;; 右下
    matrix
    )
  )

(setq init-num 1000)
(defvar *wrench-list* '())
(defun generate-circle-trajectory (&key (r-start-pos) (r-radius) (r-angle) (r-finish-angle) (l-start-pos) (l-radius) (l-angle) (l-finish-angle) (num-points) (mass))
  (setq *wrench-list* '())
  (setq grav 9.8)
  (setq force_fit 100.0) ;;?
  (let* ((r-center2hand (float-vector 0.0 (- 0.0 (* r-radius (cos r-angle))) (* r-radius (sin r-angle))))
	 (r-center (v- r-start-pos r-center2hand))
	 (l-center2hand (float-vector 0.0 (* l-radius (sin l-angle)) (* l-radius (cos l-angle))))
	 (l-center (v- l-start-pos l-center2hand))
	 (r-trajectory '())
	 (l-trajectory '())) ;; 結果を蓄積するリスト
    ;; (setq c1 (make-cube 100 100 100))
    ;; (setq c2 (make-cube 100 100 100))
    ;; (send c1 :locate center :world)
    ;;
  ;; (setq b '())
    ;;
    (dotimes (i (+ num-points 1))
      ;; (print "Enter to next")(read-line)
      ;; (let ((theta (* i theta-step)))
      ;; (setq theta (+ theta angle))
      ;; (let ((theta (theta-cos :a (/ (- angle finish-angle) 2.0)
      ;; 			      :b (/ pi num-points)
      ;; 			      :index i
      ;; 			      :c (+ (/ (- finish-angle angle) 2.0) angle)
      ;; 			      )))
      (let ((r-theta (theta-3 :x1 0.0
			      :y1 r-angle
			      :x2 (float num-points)
			      :y2 r-finish-angle
			      :index (float i)
			      ))
	    (l-theta (theta-3 :x1 0.0
			      :y1 l-angle
			      :x2 (float num-points)
			      :y2 l-finish-angle
			      :index (float i)
			      )))
	;;
;;	(push theta b)
	;;

	(setq r-target-circle (float-vector 0.0 (- 0.0 (* r-radius (cos r-theta))) (* 0.9 (* r-radius (sin r-theta)))))
	(setq r-circle (v+ r-center r-target-circle))  ;;rarm
	(setq l-target-circle (float-vector 0.0 (* 1.4 (* l-radius (sin l-theta))) (* 1.0 (* l-radius (cos l-theta)))))
	(setq l-circle (v+ l-center l-target-circle))  ;;larm

	(setq r_force (/ (* (* mass grav) (cos r-theta)) 2.0))
	;;(setq r_moment (* r_force (/ r-radius 1000)))
	(setq r_fx 0.0)
	(setq r_fy (+ (* r_force (sin r-theta)) (* force_fit (cos r-theta))))
	(setq r_fz (- (* r_force (cos r-theta)) (* force_fit (sin r-theta))))
	(setq r_fx (- r_fx))
	(setq r_fy (- r_fy))
	(setq r_fz (- r_fz))
	;; (incf r_corr 0.5)
	;; (when (< r_corr r_fy)
	;;   (setq r_fy r_corr)
	;;   )
	;; (when (< r_corr r_fz)
	;;   (setq r_fz r_corr)
	;;   )

	(setq r_Mx (/ (- (* (aref r-target-circle 1) r_fz) (* (aref r-target-circle 2) r_fy)) 1000))
	(setq r_My 0.0)
	(setq r_Mz 0.0)
	(setq r_wr (float-vector r_fx r_fy r_fz r_Mx r_My r_Mz))
	;; (setq r_wr (transform (transform-matrix :rarm t) r_wr))
	(setq l_force 50.0)
	(setq l_fx 0.0)
	(setq l_fy (- 0.0 (* l_force (sin l-theta))))
	(setq l_fz (- 0.0 (* l_force (cos l-theta))))
	(setq l_fx (- l_fx))
	(setq l_fy (- l_fy))
	(setq l_fz (- l_fz))
	;; (decf l_corr 0.2)
	;; (when (> l_corr l_fz)
	;;   (setq l_fz l_corr)
	;;   )
	
	(setq l_wr (float-vector l_fx l_fy l_fz 0.0 0.0 0.0))
	;; (setq l_wr (transform (transform-matrix :rarm nil) l_wr))
	(setq rl_wr (float-vector 0.0 0.0 0.0 0.0 0.0 0.0))
	(setq ll_wr (float-vector 0.0 0.0 0.0 0.0 0.0 0.0))
	(setq wr (concatenate float-vector rl_wr ll_wr r_wr l_wr))
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
    (setq init-list-wr '())
    (dotimes (i init-num)
      (setq ry_corr (theta-3 :x1 0.0 :y1 0.0 :x2 (float init-num) :y2 (elt (car *wrench-list*) 13) :index (float i)))
      (setq rz_corr (theta-3 :x1 0.0 :y1 0.0 :x2 (float init-num) :y2 (elt (car *wrench-list*) 14) :index (float i)))
      (setq rmx_corr (theta-3 :x1 0.0 :y1 0.0 :x2 (float init-num) :y2 (elt (car *wrench-list*) 15) :index (float i)))
      (setq lz_corr (theta-3 :x1 0.0 :y1 0.0 :x2 (float init-num) :y2 (elt (car *wrench-list*) 20) :index (float i)))
      (setq init-wr (float-vector 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ry_corr rz_corr rmx_corr 0.0 0.0 0.0 0.0 lz_corr 0.0 0.0 0.0))
      (push init-wr init-list-wr)
      )
    (nreverse init-list-wr)
    (setq *wrench-list* (append init-list-wr *wrench-list*))
    ;;
    ;; (setq a (gnuplot))
    ;; (nreverse b)
    ;; (print b)
    ;; (send a :draw b)
    ;;
    (setq trajectory '())
    (push l-trajectory trajectory) ;; elt trajectory 1
    (push r-trajectory trajectory) ;; elt trajectory 0
    trajectory
    )
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
			   (send *robot* :link-list (send *robot* :rarm :end-coords :parent))
			   (send *robot* :link-list (send *robot* :larm :end-coords :parent))
			   ;; (send *robot* :link-list (send *robot* :rarm :end-coords :parent) (send *robot* :chest_link2_lk))
			   ;; (send *robot* :link-list (send *robot* :larm :end-coords :parent) (send *robot* :chest_link2_lk))
			   )
	  :translation-axis (list t t t t) 
	  ;; :rotation-axis (list t t nil nil)
	  ;; :rotation-axis (list t t t nil)
	  :rotation-axis (list t t t t)
	  :target-centroid-pos (midpoint centroid
	  				 (send *robot* :rleg :end-coords :worldpos)
	  				 (send *robot* :lleg :end-coords :worldpos))
	  :cog-translation-axis :z
	  :thre (list 1.0 1.0 0.1 0.1)
	  :rthre (list (deg2rad 1.0) (deg2rad 1.0) 0.01 0.01)
	  )
    ;;(objects (list *robot* rarm-target))
    ;; (send *robot* :newcoords (make-coords))
    )
  )

(defvar *angle-list* '())
(defvar *time-list* '())
(defun follow-circle (&key (r_tra nil) (l_tra nil) (run nil) &allow-other-keys)
  ;;(setq r-angle 60)
  ;;(setq same-angle (/ r-angle (- (length r_tra) 1)))
  (setq start-value 0.2)
  (setq finish-value 0.8)
  (setq diff (/ (- finish-value start-value) (- (length r_tra) 1)))
  (setq l-start 90.0)
  (setq l-diff (/ l-start (- (length r_tra) 1)))
  (setq r-pitch -30.0)
  (setq r-yaw 30.0)
  ;; (setq r-pitch -50)
  ;; (setq r-yaw 42)
  (setq pitch-diff (/ l-diff 1))  ;1or2..
  (setq r-diff (* (/ l-diff 3) 2))

  (setq cop -0.1)
  (setq cop-diff (/ (- (- cop) cop) (length r_tra)))
  (setq cop-list '())
  
  (setq *angle-list* '())
  (setq *time-list* '())
  (send *ri* :set-interpolation-mode :linear)
  ;; (dolist (target r_tra)
  (progn (reset-pose :run nil) (lleg-foot :run nil) (lower-waist :run nil) (start-pos :run nil))
  (setq init-angle (send *robot* :angle-vector))
  (dotimes (i (length r_tra))
    ;;(arm-solve-ik-locate :rarm-trans-pos target :rarm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad r-angle)))
    (setq r_target (elt r_tra i))
    (setq l_target (elt l_tra i))
    ;; (progn (reset-pose :run nil) (lleg-foot :run nil) (lower-waist :run nil) (start-pos :run nil))
    ;; (send *robot* :angle-vector #f(3.09242 -6.86101 -43.9995 56.6436 -15.467 0.272271 3.67336 4.96019 -36.3772 44.6703 -11.1064 -11.5631 8.46561 24.3388 16.4479 0.0 0.0 46.1851 5.25702 -76.4537 -12.7614 -28.6211 -37.8201 70.1417 -65.2121 -28.4953 -6.69292 29.8917 15.5053 -88.4715 -46.2998 13.644 -2.36926 0.0 90.0 0.0 0.0 0.0 0.0 0.0 90.0 0.0 0.0 0.0 0.0))
    ;; (send *robot* :fix-leg-to-coords (make-coords))
    ;; (send *robot* :angle-vector init-angle)
    (full-solve-ik-locate :rarm-trans-pos r_target
    			  :larm-trans-pos l_target
    			  ;; :rarm-rpy (float-vector (deg2rad 60) (deg2rad -30) (deg2rad 0))
    			  ;; :rarm-rpy (float-vector (deg2rad 60) (deg2rad r-pitch) (deg2rad r-yaw)) ;;pattern2
    			  :rarm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad l-start))  ;;pattern1
    			  :larm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad l-start))
    			  :centroid start-value)
    ;; (full-link6-ik-locate :rarm-trans-pos r_target
    ;; 			  :larm-trans-pos l_target
    ;; 			  ;; :rarm-rpy (float-vector (deg2rad 41) (deg2rad r-pitch) (deg2rad r-yaw))
    ;; 			  :larm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad l-start))
    ;; 			  :centroid start-value)
    ;;(objects (list *robot* r_target))
    ;;(print "Enter to next")(read-line)
    ;;(unix:usleep 1000)
    ;; (when run
    ;;   (send *ri* :angle-vector (send *robot* :angle-vector) 500)
    ;;   (send *ri* :wait-interpolation))
    (setq start-value (+ start-value diff))
    ;;(setq r-angle (- r-angle same-angle))
    (setq l-start (- l-start l-diff))
    (setq r-pitch (+ r-pitch pitch-diff))
    (setq r-yaw (- r-yaw r-diff))

    (push cop cop-list)
    (setq cop (+ cop cop-diff))
    
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
  ;; (setq a 10.0)
  ;; (dotimes (i 499)
  ;;   (setf (elt *time-list* i) a)
  ;;   (setq a (- a 0.018))
  ;;   )

  (nreverse cop-list)
  
  (dotimes (i init-num)
    (push init-angle *angle-list*)
    (push 2.0 *time-list*)
    )
  
  (when run
    (send *ri* :angle-vector (car *angle-list*))
    (send *ri* :wait-interpolation)
    (send *ri* :angle-vector-sequence-full *angle-list* *time-list* :wrench *wrench-list*)
    (send *ri* :change-cop 0.0 0.1 0.0 0.1 12.0)  ;; default rleg_x 0.0 rleg_y 0.02 lleg_x 0.0 lleg_y -0.02 t 2.0
    ;; (send *ri* :set-auto-stabilizer-param :default-zmp-offsets (list #f(0 0.02) #f(0 -0.02)))
    ;; (unix:sleep 0.5)
    ;; (send *ri* :set-auto-stabilizer-param :default-zmp-offsets (list #f(0 0.02) #f(0 -0.02)))
    ;; (send *ri* :wait-interpolation)
    )
  )


(defun wake (&rest args)  
  ;; (apply #'follow-circle (generate-circle-trajectory (v+ #f(0 400 0) (send *robot* :rarm :end-coords :worldpos)) 400 10) args)
  (setq tra (generate-circle-trajectory :r-start-pos (send *robot* :rarm :end-coords :worldpos) :r-radius 500.0 :r-angle (deg2rad 0) :r-finish-angle (deg2rad 90) :l-start-pos (send *robot* :larm :end-coords :worldpos) :l-radius 200.0 :l-angle (deg2rad 0) :l-finish-angle (deg2rad 90) :num-points 5000 :mass 20.0))
  (apply #'follow-circle (append args (list :r_tra (elt tra 0) :l_tra (elt tra 1))))  ;;larm200or275
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

(defun wrench-drawplot (&key (list *wrench-list*) (num 13))
  (setq a (gnuplot))
  (setq b '())
  (dotimes (i (length list))
    (push (elt (elt list i) num) b)
    )
  (nreverse b)
  (send a :draw b)
  )

(defun wake2
    (&key(transpos #f(0 0 0)))

  ;; 動かす位置を右手のlink6に設定
  (setq move-target-coords (arrow))
  (send move-target-coords :newcoords (send *robot* :rarm_link6_lk :copy-worldcoords))

  ;; 目標位置の設定
  (setq target-coords (send
  		       (send (send *robot* :rarm_link6_lk :copy-worldcoords))
  		       :translate transpos :world))

  ;; assoc
  (send (send (send *robot* :rarm :end-coords) :parent) :assoc move-target-coords)
  
  (send *robot* :rarm :inverse-kinematics target-coords
  	:move-target move-target-coords)
  
  (setq rarm_link6 (send *robot* :rarm_link6_lk :copy-worldcoords))
  (format t "after rarm-link6: ~A~%" rarm_link6)

  ;; (objects (list *robot* rarm_link6 move-target-coords target-coords (send *robot* :rarm :end-coords :copy-worldcoords)))
  (objects (list *robot* target-coords move-target-coords))
  )

;;(send *ri* :stop-impedance :larm)
;;progn (reset-pose :run nil) (lleg-foot :run nil) (lower-waist :run nil) (start-pos :run nil) (wake)
;;progn (reset-pose) (lleg-foot :run nil) (send *ri* :angle-vector (send *robot* :angle-vector) 5000) (lower-waist) (start-pos) (wake :run t)
;;;;;;;;;;;;

;; (defun test-motion ()
;;   (send *robot* :reset-pose)
;;   (lleg-foot)
;;   (initial-pos)
;;   (bed-knee)
;;   (support)
;;   (lower-waist)
;;   (rarm-neck)
;;   (let ((traj (generate-circle-trajectory (v+ #f(0 200 0) (send *robot* :rarm :end-coords :worldpos)) 200 10)))
;;     (dolist (pos traj)
;;       (setq cube (make-cube 100 100 100))
;;       (send cube :translate pos)
;;       (full-solve-ik-locate :rarm-trans-pos pos)
;;       (objects (list *robot* cube))
;;       (send *ri* :angle-vector (send *robot* :angle-vector))
;;       (send *ri* :wait-interpolation)
;;       (read-line)
;;       )
;;     )
;;   )

;; (defun rotation-matrix-to-rpy (rot-matrix)
;;   (let* ((r11 (aref rot-matrix 0 0))
;;          (r12 (aref rot-matrix 0 1))
;;          (r13 (aref rot-matrix 0 2))
;;          (r21 (aref rot-matrix 1 0))
;;          (r22 (aref rot-matrix 1 1))
;;          (r23 (aref rot-matrix 1 2))
;;          (r31 (aref rot-matrix 2 0))
;;          (r32 (aref rot-matrix 2 1))
;;          (r33 (aref rot-matrix 2 2))
;;          (yaw (atan2 r21 r11))
;;          (pitch (atan2 (- r31) (sqrt (+ (* r11 r11) (* r21 r21)))))
;;          (roll (atan2 r32 r33)))
;;     (list yaw pitch roll)))

;; (defun full-link6-ik-locate
;;     (&key(rarm-trans-pos (send *robot* :rarm_link6_lk :worldpos))
;; 	 (larm-trans-pos (send *robot* :larm :end-coords :worldpos))
;; 	 (rarm-rpy (rotation-matrix-to-rpy (send *robot* :rarm_link6_lk :worldrot)))
;; 	 (larm-rpy (rotation-matrix-to-rpy (send *robot* :larm :end-coords :worldrot)))
;; 	 (centroid 0.5))
;;   (let* ((rleg-coords (send *robot* :rleg :end-coords :copy-worldcoords))
;; 	 (lleg-coords (send *robot* :lleg :end-coords :copy-worldcoords))
;; 	 (rarm-link6-coords (send *robot* :rarm_link6_lk :copy-worldcoords))
;; 	 (larm-link6-coords (send *robot* :larm :end-coords :copy-worldcoords))
;; 	 (rarm-translation-coords (send rarm-link6-coords :locate rarm-trans-pos :world) :copy-worldcoords)
;; 	 (larm-translation-coords (send larm-link6-coords :locate larm-trans-pos :world) :copy-worldcoords)
;; 	 (rarm-target (make-coords :pos (send rarm-translation-coords :pos)
;; 				   :rpy rarm-rpy))
;; 	 (larm-target (make-coords :pos (send larm-translation-coords :pos)
;; 				   :rpy larm-rpy)))
;;     (send *robot* :fullbody-inverse-kinematics
;; 	  (list rleg-coords lleg-coords rarm-target larm-target)
;; 	  :move-target (list (send *robot* :rleg :end-coords)
;; 			     (send *robot* :lleg :end-coords)
;; 			     (send *robot* :rarm_link6_lk)
;; 			     (send *robot* :larm :end-coords))
;; 	  :link-list (list (send *robot* :link-list (send *robot* :rleg :end-coords :parent))
;; 			   (send *robot* :link-list (send *robot* :lleg :end-coords :parent))
;; 			   (send *robot* :link-list (send *robot* :rarm :end-coords :parent))
;; 			   (send *robot* :link-list (send *robot* :larm :end-coords :parent)))
;; 	  :translation-axis (list t t t t) 
;; 	  ;; :rotation-axis (list t t nil nil)
;; 	  ;; :rotation-axis (list t t t nil)
;; 	  :rotation-axis (list t t t t)
;; 	  :target-centroid-pos (midpoint centroid
;; 	  				 (send *robot* :rleg :end-coords :worldpos)
;; 	  				 (send *robot* :lleg :end-coords :worldpos))
;; 	  :cog-translation-axis :z
	  
	  
;; 	  )
;;     )
;;   )

;;progn (reset-pose :run nil) (lleg-foot :run nil) (lower-waist :run nil) (start-pos :run nil)

;;progn (reset-pose :run nil) (lleg-foot :run nil) (lower-waist :run nil)
;;irt-ri
;;send *ri* :start-auto-balancer
;;start-pos
;;wake :run t
