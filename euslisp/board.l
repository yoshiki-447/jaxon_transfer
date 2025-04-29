;; (load "package://eus_prioritized_inverse_kinematics/euslisp/prioritized-inverse-kinematics-compiled.l")
(require "models/arrow-object.l")

(defun arm-solve-ik
    (&key(rarm-trans-pos #f(0 0 0))
	 (larm-trans-pos #f(0 0 0))
	 (rarm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad 0)))
	 (larm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad 0))))
  (let* ((rarm-end-coords (send *robot* :rarm :end-coords :copy-worldcoords))
	 (larm-end-coords (send *robot* :larm :end-coords :copy-worldcoords))
	 ;; (rarm-translation-coords (send rarm-end-coords :translate rarm-trans-pos :world) :copy-worldcoords)
	 ;; (larm-translation-coords (send larm-end-coords :translate larm-trans-pos :world) :copy-worldcoords)
	 ;; (rarm-target (make-coords :pos (send rarm-translation-coords :pos)
	 ;; 			   :rpy rarm-rpy))
	 ;; (larm-target (make-coords :pos (send larm-translation-coords :pos)
	 ;; 			   :rpy larm-rpy)))
	 (rarm-target (send rarm-end-coords :translate rarm-trans-pos :world) :copy-worldcoords)
	 (larm-target (send larm-end-coords :translate larm-trans-pos :world) :copy-worldcoords))
    (send *robot* :inverse-kinematics (list rarm-target larm-target)
	  :move-target (list (send *robot* :rarm :end-coords)
			     (send *robot* :larm :end-coords))
	  :link-list (list (send *robot* :link-list (send *robot* :rarm :end-coords :parent))
			   (send *robot* :link-list (send *robot* :larm :end-coords :parent)))
	  :translation-axis (list t t)
	  :rotation-axis (list t t)
	  :null-space #f(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
	  :thre (list 2.0 2.0)
	  :rthre (list 0.1 0.1)
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
	  :link-list (list ;; (send *robot* :link-list (send *robot* :rarm :end-coords :parent))
			   (send *robot* :link-list (send *robot* :rarm :end-coords :parent) (send *robot* :chest_link2_lk))
			   ;; (send *robot* :link-list (send *robot* :larm :end-coords :parent)))
			   (send *robot* :link-list (send *robot* :larm :end-coords :parent) (send *robot* :chest_link2_lk)))
	  :translation-axis (list t t)
	  :rotation-axis (list t t)
	  ;; :target-centroid-pos (midpoint 0.5
	  ;; 				 (send *robot* :rleg :end-coords :worldpos)
	  ;; 				 (send *robot* :rleg :end-coords :worldpos))
	  ;; :cog-translation-axis :z
	  ;; :thre (list 2.0 2.0)
	  ;; :rthre (list 1.0 1.0)
	  )
    )
  )

(defun zmp-start ()
  (send *ri* :start-auto-balancer)
  (send *ri* :set-auto-stabilizer-param :default-zmp-offsets '(#f(0.05 0.02) #f(0.05 -0.02)))
  )

(defun lleg-foot (&key (run t))
  (send *robot* :reset-pose)
  (send *robot* :fix-leg-to-coords (make-coords))
  (let ((target-lleg-coords1 (send
			      (send
			       (send *robot* :lleg :end-coords :copy-worldcoords)
			       :translate (float-vector 0 100 0))
			      :rotate (deg2rad 25) :z)
			     )
	(target-lleg-coords2 (send
			      (send
			       (send *robot* :lleg :end-coords :copy-worldcoords)
			       :translate (float-vector 0 200 0))  ;;200 z45
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
			       :translate (float-vector 0 -200 0)) ;;-200z-45
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
	   (send *robot* :lleg :inverse-kinematics target-lleg-coords1)
	   (send *robot* :lleg :inverse-kinematics target-lleg-coords2)
	   (send *robot* :rleg :inverse-kinematics target-rleg-coords1)
	   (send *robot* :rleg :inverse-kinematics target-rleg-coords2)
	   ;;(send *robot* :angle-vector (send *ri* :state :potentio-vector))
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

(defun fix-coords ()  ;;after lower-waist
  (setq a (make-coords))
  (send a :translate (float-vector (aref (send (send *robot* :foot-midcoords) :pos) 0) (aref (send (send *robot* :foot-midcoords) :pos) 1) 0))  ;;lower-waist z しても０にするため
  (send a :rotate (deg2rad 45) :z) 
  (send *robot* :fix-leg-to-coords a)
  (format t "a: ~A~%" a)
  )

(defun move-waist (&key(dz) (tm) (run t))
       (send *robot* :fix-leg-to-coords (make-coords))
       (send *robot* :legs :move-end-pos (float-vector 0 0 dz))
       (send *robot* :move-centroid-on-foot :both '(:rleg :lleg))
       (when run
	 (send *ri* :angle-vector (send *robot* :angle-vector) tm)
	 (send *ri* :wait-interpolation)
	 )
       ;;(fix-coords)
       (send *robot* :newcoords (make-coords))
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

(defun reset-pose (&key(run t))
  (send *robot* :reset-pose)
  (when run
    (send *ri* :change-cop 0.0 0.02 0.0 -0.02 5.0)
    (send *ri* :angle-vector (send *robot* :angle-vector) 5000)
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

(defun link-end (&key(link-name :rarm_link0_lk))  ;;rarm-end-coordsとのxy距離
  (setq collar (float-vector (aref (send *robot* link-name :worldpos) 0) (aref (send *robot* link-name :worldpos) 1) (aref (send *robot* :rarm :end-coords :worldpos) 2)))
  (setq collar2end (v- (send *robot* :rarm :end-coords :worldpos) collar))
  (norm collar2end)
  )

(defun manequine (&key (rotate nil))
  (send *human* :move-to (make-coords :pos #f(525 -375 900) :rpy (float-vector (deg2rad 150) (deg2rad 0) (deg2rad 0))))
  (send *human* :rleg :crotch-p :joint-angle -60)
  (send *human* :rleg :knee-p :joint-angle 60)
  (send *human* :lleg :crotch-p :joint-angle -60)
  (send *human* :lleg :knee-p :joint-angle 60)
  (send *human* :rarm :elbow-p :joint-angle -120)
  (send *human* :rarm :shoulder-p :joint-angle -20)
  (send *human* :rarm :shoulder-y :joint-angle -60)
  (send *human* :larm :elbow-p :joint-angle -120)
  (send *human* :larm :shoulder-p :joint-angle -20)
  (send *human* :larm :shoulder-y :joint-angle -60)
  
  (when rotate
    (send *human* :move-to (make-coords :rpy (float-vector (deg2rad 70) 0 0)))
    (send *human* :locate #f(500 350 900) :world)
    )
  )

(defun posture (&key(run t))
  ;;(send *robot* :reset-pose)
  (send *robot* :torso :waist-p :joint-angle 25)
  ;; (full-solve-ik-locate :rarm-trans-pos #f(500 -600 775) :larm-trans-pos #f(775 -300 1050) 
  ;; 			:rarm-rpy (float-vector (deg2rad -40) (deg2rad 0) (deg2rad 20))
  ;; 			:larm-rpy (float-vector (deg2rad -35) (deg2rad 0) (deg2rad 0))
  ;; 			:centroid 0.2)  ::rarm 1.15
  (full-solve-ik-locate :rarm-trans-pos #f(450 -550 775) :larm-trans-pos #f(725 -250 1050) 
  			:rarm-rpy (float-vector (deg2rad -40) (deg2rad 0) (deg2rad 20))
  			:larm-rpy (float-vector (deg2rad -35) (deg2rad 0) (deg2rad 0))
  			:centroid 0.2)  ;;rarm 1.2
  ;; (full-solve-ik-locate :rarm-trans-pos #f(400 -550 775) :larm-trans-pos #f(600 -250 1050) 
  ;; 			:rarm-rpy (float-vector (deg2rad -30) (deg2rad 0) (deg2rad 20))
  ;; 			:larm-rpy (float-vector (deg2rad -30) (deg2rad 0) (deg2rad 0))
  ;; 			:centroid 0.2)
  ;;(send *robot* :rarm :shoulder-p :joint-angle -10)
  
  ;; (send *robot* :torso :waist-y :joint-angle -25)
  ;; (send *robot* :torso :waist-p :joint-angle 30)  ;;25
  ;; (full-solve-ik-locate :larm-trans-pos #f(700 -350 1050) :rarm-trans-pos #f(450 -700 775)
  ;; 			:rarm-rpy (float-vector (deg2rad -35) (deg2rad 0) (deg2rad 20))
  ;; 			:larm-rpy (float-vector (deg2rad -35) (deg2rad 0) (deg2rad 0))
  ;; 			:centroid 0.2)  ;;human pattern1,2
  ;; (full-solve-ik-locate :larm-trans-pos #f(700 -350 1050) :rarm-trans-pos #f(450 -700 775)
  ;; 			:rarm-rpy (float-vector (deg2rad -35) (deg2rad 0) (deg2rad 0))
  ;; 			:larm-rpy (float-vector (deg2rad -35) (deg2rad 0) (deg2rad 0))
  ;; 			:centroid 0.2)
  ;; (full-solve-ik-locate :larm-trans-pos #f(650 -300 1100) :rarm-trans-pos #f(400 -650 825)
  ;; 			:rarm-rpy (float-vector (deg2rad -35) (deg2rad 0) (deg2rad 0))
  ;; 			:larm-rpy (float-vector (deg2rad -35) (deg2rad 0) (deg2rad 0))
  ;; 			:centroid 0.2) ;;右手左手手前
  ;; (full-solve-ik-locate :larm-trans-pos #f(700 -350 1000) :rarm-trans-pos #f(400 -650 725)
  ;; 			:rarm-rpy (float-vector (deg2rad -35) (deg2rad 30) (deg2rad 0))
  ;; 			:larm-rpy (float-vector (deg2rad -35) (deg2rad 30) (deg2rad 0))
  ;; 			:centroid 0.2) ;;add jointangle
  ;; (send *robot* :rarm :wrist-p :joint-angle -40)
  ;; (send *robot* :larm :wrist-p :joint-angle 0)
  ;; (prioritized-ik :relbow-trans-pos #f(-50 -50 -50) :relbow-trans-rpy (float-vector (deg2rad 60) (deg2rad 0) (deg2rad 0))
  ;; 		  :lelbow-trans-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad -30)))
  (when run
    (send *ri* :change-cop 0.02 -0.20 0.02 -0.20 5.0)
    (send *ri* :wait-interpolation)
    (send *ri* :angle-vector (send *robot* :angle-vector) 10000)
    (send *ri* :wait-interpolation)
    )
  )

(defun all ()
  (send *ri* :start-auto-balancer)
  (lleg-foot)
  (lower-waist)
  (posture)
  (impedance)
  (rotate :run t)
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

(defun minimum-leap (&key (x1) (y1) (x2) (y2) (index))
  (setq mat (matrix (list (expt x1 5) (expt x1 4) (expt x1 3) (expt x1 2) x1 1)
		    (list (expt x2 5) (expt x2 4) (expt x2 3) (expt x2 2) x2 1)
		    (list (* 5 (expt x1 4)) (* 4 (expt x1 3)) (* 3 (expt x1 2)) (* 2 x1) 1 0)
		    (list (* 5 (expt x2 4)) (* 4 (expt x2 3)) (* 3 (expt x2 2)) (* 2 x2) 1 0)
		    (list (* 20 (expt x1 3)) (* 12 (expt x1 2)) (* 6 x1) 2 0 0)
		    (list (* 20 (expt x2 3)) (* 12 (expt x2 2)) (* 6 x2) 2 0 0)
		    ))
  (setq vec (float-vector y1 y2 0.0 0.0 0.0 0.0))  ;;位置、速度、加速度
  (setq coeffs (transform (inverse-matrix mat) vec))
  (setq ax5 (* (aref coeffs 0) (expt index 5)))
  (setq bx4 (* (aref coeffs 1) (expt index 4)))
  (setq cx3 (* (aref coeffs 2) (expt index 3)))
  (setq dx2 (* (aref coeffs 3) (expt index 2)))
  (setq ex (* (aref coeffs 4) index))
  (setq f (aref coeffs 5))
  (setq y (+ ax5 bx4 cx3 dx2 ex f))
  y
  )

(defvar *r-theta-list* '())
(defvar *l-theta-list* '())
(setq init-num 1000)
(defvar *wrench-list* '())
(defun generate-circle-trajectory (&key (r-start-pos) (r-angle) (r-finish-angle) (r-height) (l-start-pos) (l-angle) (l-finish-angle) (l-height) (num-points) (mass))
  (setq *wrench-list* '())
  (setq cube-list '())
  (setq myu 0.5)
  (setq grav 9.8)
  ;;(setq force_fit 30.0) ;;?
  (let* (;; (r-center2hand (float-vector (aref (send *robot* :rarm :end-coords :worldpos) 0) (aref (send *robot* :rarm :end-coords :worldpos) 1) 0.0))
	 ;; (r-center (v- r-start-pos r-center2hand))
	 (r-center (float-vector (aref (send *robot* :chest_link0_lk :worldpos) 0) (aref (send *robot* :chest_link0_lk :worldpos) 1) (aref (send *robot* :rarm :end-coords :worldpos) 2)))
	 (r-center2hand (v- (send *robot* :rarm :end-coords :worldpos) r-center))
	 (r-radius (norm r-center2hand))
	 ;; (r-diff-z (/ r-height num-points))
	 ;; (r-ini-theta (atan (aref (send *robot* :rarm :end-coords :worldpos) 0) (aref (send *robot* :rarm :end-coords :worldpos) 1)))
	 ;; (r-ini-theta (atan (aref r-center2hand 0) (aref r-center2hand 1)))
	 
	 (r-ini-theta (atan (aref r-center2hand 1) (aref r-center2hand 0)))
	 
	 ;; (l-center2hand (float-vector (aref (send *robot* :larm :end-coords :worldpos) 0) (aref (send *robot* :larm :end-coords :worldpos) 1) 0.0))
	 ;; (l-center (v- l-start-pos l-center2hand))
	 ;; (l-center (float-vector 0.0 0.0 (aref (send *robot* :larm :end-coords :worldpos) 2)))
	 (l-center (float-vector (aref (send *robot* :chest_link0_lk :worldpos) 0) (aref (send *robot* :chest_link0_lk :worldpos) 1) (aref (send *robot* :larm :end-coords :worldpos) 2)))
	 (l-center2hand (v- (send *robot* :larm :end-coords :worldpos) l-center))
	 (l-radius (norm l-center2hand))
	 ;; (l-diff-z (/ l-height num-points))
	 ;; (setq ini-theta (atan (aref (send *robot* :larm :end-coords :worldpos) 0) (aref (send *robot* :larm :end-coords :worldpos) 1)))
	 ;; (l-ini-theta (atan (aref l-center2hand 0) (aref l-center2hand 1)))
	 
	 (l-ini-theta (atan (aref l-center2hand 1) (aref l-center2hand 0)))
	 ;;(setq theta-step (/ (- (/ pi 2) angle) num-points))
	 ;; (r-theta-step (/ (- (deg2rad 60) r-angle) num-points))  ;;60or90
	 ;; (l-theta-step (/ (- (deg2rad 60) l-angle) num-points))  ;;60or90
	 (r-trajectory '())
	 (l-trajectory '()))
    ;; (setq c1 (make-cube 100 100 100))
    ;; (setq c2 (make-cube 100 100 100))
    ;; (send c1 :locate center :world)
    (setq *r-theta-list* '())
    (setq *l-theta-list* '())
    (dotimes (i (+ num-points 1))
      ;; (print "Enter to next")(read-line)
      (let ((r-theta (minimum-leap :x1 0.0
			      :y1 r-angle
			      :x2 (float num-points)
			      :y2 r-finish-angle
			      :index (float i)))
	    ;; (r-theta (theta-3 :x1 0.0
	    ;; 		      :y1 r-angle
	    ;; 		      :x2 (float num-points)
	    ;; 		      :y2 r-finish-angle
	    ;; 		      :index (float i)))
	    (l-theta (minimum-leap :x1 0.0
			      :y1 l-angle
			      :x2 (float num-points)
			      :y2 l-finish-angle
			      :index (float i)))
	    )
	;; (r-diff-z (* i r-diff-z)) (l-diff-z (* i l-diff-z))
	(cond ((< i (/ num-points 2))
	       (setq rhand-z (minimum-leap :x1 0.0
					   :y1 0.0
					   :x2 (float (/ num-points 2))
					   :y2 r-height
					   :index (float i)))
	       (setq lhand-z (minimum-leap :x1 0.0
					   :y1 0.0
					   :x2 (float (/ num-points 2))
					   :y2 l-height
					   :index (float i)))
	       )
	      (t
	       (setq rhand-z r-height)
	       (setq lhand-z l-height)
	       )
	      )
	
	;; (setq r-theta (- r-ini-theta r-theta))
	;; (setq l-theta (- l-ini-theta l-theta))
	
	;; (setq theta (+ theta angle))

	(push (rad2deg r-theta) *r-theta-list*)
	(push (rad2deg l-theta) *l-theta-list*)
	
	;; (setq r-target-circle (float-vector (* 1.0 (* r-radius (sin r-theta))) (* 1.0 (* r-radius (cos r-theta))) rhand-z))
	;; (setq r-circle (v+ r-center r-target-circle))
	;; (setq l-target-circle (float-vector (* 1.0 (* l-radius (sin l-theta))) (* 1.0 (* l-radius (cos l-theta))) lhand-z))
	;; (setq l-circle (v+ l-center l-target-circle))
	
	
	(setq r-target-circle (rotate-vector (float-vector (* 1.0 (* r-radius (cos r-theta))) (* 1.2 (* r-radius (sin r-theta))) rhand-z) r-ini-theta :z))
	(setq r-circle (v+ r-center r-target-circle))
	(setq l-target-circle (rotate-vector (float-vector (* 1.0 (* l-radius (cos l-theta))) (* 1.0 (* l-radius (sin l-theta))) lhand-z) l-ini-theta :z))
	(setq l-circle (v+ l-center l-target-circle))
	
	;;(setq r-circle (v+ (send r-center :pos) (send r-target-circle :pos)));;rarm
	;;(setq l-circle (v+ (send l-center :pos) (send l-target-circle :pos)))  ;;larm
	(setq r-wr-theta (+ r-theta r-ini-theta))
	(setq l-wr-theta (+ l-theta l-ini-theta))
	(setq r_force (* myu mass grav))
	;;(setq r_moment (* r_force (/ r-radius 1000)))
	;; (setq r_fx (* r_force (cos (- pi r-theta))))
	;; (setq r_fy (* r_force (sin (- pi r-theta))))
	(setq r_fx (* r_force (- (sin r-wr-theta))))
	(setq r_fy (* r_force (cos r-wr-theta)))
	(setq r_fz 50.0)
	(setq r_fx (- r_fx))
	(setq r_fy (- r_fy))
	(setq r_fz (- r_fz))
	
	(setq r_Mx 0.0)
	(setq r_My 0.0)
	;; (setq r_Mz (/ (- (* (aref r-target-circle 0) r_fy) (* (aref r-target-circle 1) r_fx)) 1000))
	(setq r_Mz 0.0)
	(setq r_wr (float-vector r_fx r_fy r_fz r_Mx r_My r_Mz))
	(setq l_force 50.0)
	;; (setq l_fx (* l_force (cos l-theta)))
	;; (setq l_fy (- 0.0 (* l_force (sin l-theta))))
	(setq l_fx (* l_force (sin l-wr-theta)))
	(setq l_fy (- 0.0 (* l_force (cos l-wr-theta))))
	(setq l_fz 0.0)
	(setq l_fx (- l_fx))
	(setq l_fy (- l_fy))
	(setq l_fz (- l_fz))
	(setq l_wr (float-vector l_fx l_fy l_fz 0.0 0.0 0.0))
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
      (when (= (mod i 100) 0)
      (setq a (make-cube 20 20 20))
      (send a :set-color :red) ;;rarm
      (send a :translate r-circle)
      (push a cube-list)
      (setq b (make-cube 20 20 20))
      (send b :set-color :blue) ;;larm
      (send b :translate l-circle)
      (push b cube-list)
      )
      )
    (nreverse *r-theta-list*)
    (nreverse *l-theta-list*)
    
    ;; (setq init-list-wr '())
    ;; (dotimes (i init-num)
    ;;   (setq rx_corr (minimum-leap :x1 0.0 :y1 0.0 :x2 (float init-num) :y2 (elt (car *wrench-list*) 12) :index (float i)))
    ;;   (setq ry_corr (minimum-leap :x1 0.0 :y1 0.0 :x2 (float init-num) :y2 (elt (car *wrench-list*) 13) :index (float i)))
    ;;   (setq rmz_corr (minimum-leap :x1 0.0 :y1 0.0 :x2 (float init-num) :y2 (elt (car *wrench-list*) 17) :index (float i)))
    ;;   (setq lx_corr (minimum-leap :x1 0.0 :y1 0.0 :x2 (float init-num) :y2 (elt (car *wrench-list*) 18) :index (float i)))
    ;;   (setq ly_corr (minimum-leap :x1 0.0 :y1 0.0 :x2 (float init-num) :y2 (elt (car *wrench-list*) 19) :index (float i)))
    ;;   (setq init-wr (float-vector 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 rx_corr ry_corr 0.0 0.0 0.0 rmz_corr lx_corr ly_corr 0.0 0.0 0.0 0.0))
    ;;   (push init-wr init-list-wr)
    ;;   )
    ;; (nreverse init-list-wr)
    ;; (setq *wrench-list* (append init-list-wr *wrench-list*))  ;;this
    (setq coords (arrow))
    (push coords cube-list)
    (push *robot* cube-list)
    (push *human* cube-list)
    (objects cube-list)
    
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
  ;; (setq r-yaw -35.0)
  ;; (setq l-yaw -35.0)
  (setq *angle-list* '())
  (setq *time-list* '())
  (send *ri* :set-interpolation-mode :linear)
  ;; (setq hand-diff (/ 90 (- (length r_tra) 1)))
  ;; (setq hand-diff (/ 70.0 (- (length r_tra) 1)))  ;;70
  ;; (setq l-hand-diff (/ 70.0 (- (length l_tra) 1))) ;;20

  ;; (progn (reset-pose :run nil) (lleg-foot :run nil) (lower-waist :run nil) (posture :run nil))
  ;; (setq init-angle (send *robot* :angle-vector))
  ;; (when run
  ;;   (send *robot* :angle-vector (send *ri* :state :potentio-vector))
  ;;   )
  (setq init-r-yaw (rad2deg (car (car (send (send *robot* :rarm :end-coords :copy-worldcoords) :rpy-angle)))))
  (setq init-l-yaw (rad2deg (car (car (send (send *robot* :larm :end-coords :copy-worldcoords) :rpy-angle)))))
  (setq hand-rotate 70)
  (dotimes (i (length r_tra))
    ;;(arm-solve-ik-locate :rarm-trans-pos target :rarm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad r-angle)))
    (setq r_target (elt r_tra i))
    (setq l_target (elt l_tra i))
    (setq r-yaw (minimum-leap :x1 0.0
			 :y1 init-r-yaw
			 :x2 (- (length r_tra) 1)
			 :y2 (+ init-r-yaw hand-rotate)
			 :index (float i)))
    (setq l-yaw (minimum-leap :x1 0.0
			 :y1 init-l-yaw
			 :x2 (- (length l_tra) 1)
			 :y2 (+ init-l-yaw hand-rotate)
			 :index (float i)))
    ;; (progn (reset-pose :run nil) (lleg-foot :run nil) (lower-waist :run nil) (posture :run nil))
    ;; (arm-solve-ik-locate :rarm-trans-pos r_target
    ;; 			  :larm-trans-pos l_target
    ;; 			  :rarm-rpy (float-vector (deg2rad r-yaw) (deg2rad 0) (deg2rad 20))
    ;; 			  ;;:rarm-rpy (float-vector (deg2rad 60) (deg2rad r-pitch) (deg2rad r-yaw)) ;;pattern2
    ;; 			  ;; :rarm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad (- l-start 90)))  ;;pattern1
    ;; 			  :larm-rpy (float-vector (deg2rad l-yaw) (deg2rad 0) (deg2rad 0))
    ;; 			  );;:centroid start-value)
    (full-solve-ik-locate :rarm-trans-pos r_target
    			  :larm-trans-pos l_target
    			  :rarm-rpy (float-vector (deg2rad r-yaw) (deg2rad 0) (deg2rad 20))
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
    ;; (setq r-yaw (+ r-yaw hand-diff))
    ;; (setq l-yaw (+ l-yaw l-hand-diff))
    
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

  ;; (dotimes (i init-num)
  ;;   ;; (push init-angle *angle-list*)
  ;;   (push (car *angle-list*) *angle-list*)
  ;;   (push 2.0 *time-list*)
  ;;   ) ;;this

  (setq start-num 200)
  
  (when run
    ;; (send *ri* :angle-vector (car *angle-list*))
    ;; (send *ri* :wait-interpolation)  ;;this
    ;; (format t "enter~%")
    ;; (read-line)
    (send *ri* :start-log)
    (send *ri* :wait-interpolation)
    (send *ri* :angle-vector (elt *angle-list* start-num) 5000)
    (send *ri* :set-ref-forces
    	(list
    	 (subseq (elt *wrench-list* start-num) 0 3) ;;rleg
    	 (subseq (elt *wrench-list* start-num) 6 9) ;;lleg
    	 (subseq (elt *wrench-list* start-num) 12 15) ;;rarm (x, y, z) 単位はニュートン
    	 (subseq (elt *wrench-list* start-num) 18 21) ;;larm
    	 )
    	5000)
    (send *ri* :wait-interpolation)
    ;;(send *ri* :angle-vector-sequence-full *angle-list* *time-list* :wrench *wrench-list*)  ;;this
    (send *ri* :angle-vector-sequence-full (nthcdr start-num *angle-list*) (nthcdr start-num *time-list*)
  	  :wrench (nthcdr start-num *wrench-list*))
    (send *ri* :change-cop 0.02 0.0 0.02 0.0 10.0)
    (send *ri* :wait-interpolation)
    )
  )


(defun rotate (&rest args)
  ;; (setq tra (generate-circle-trajectory :r-start-pos (send *robot* :rarm :end-coords :worldpos)
  ;; 					:r-angle (deg2rad 0) :r-finish-angle (deg2rad 60) :r-height 50.0
  ;; 					:l-start-pos (send *robot* :larm :end-coords :worldpos)
  ;; 					:l-angle (deg2rad 0) :l-finish-angle (deg2rad 60) :l-height 50.0
  ;; 					:num-points 5000 :mass 20.0))
  ;; (apply #'follow-circle (append args (list :r_tra (elt tra 0) :l_tra (elt tra 1))))
  (multiple-value-setq (r-tra l-tra) (generate-circle-trajectory :r-start-pos (send *robot* :rarm :end-coords :worldpos)
					:r-angle (deg2rad 0) :r-finish-angle (deg2rad 60) :r-height 30.0
					:l-start-pos (send *robot* :larm :end-coords :worldpos)
					:l-angle (deg2rad 0) :l-finish-angle (deg2rad 60) :l-height 0.0
					:num-points 5000 :mass 25.0))
  (apply #'follow-circle (append args (list :r_tra r-tra :l_tra l-tra)))
  )

(defun impedance ()
  (send *ri* :set-impedance-controller-param :arms :K-r 200.0 :D-r 300.0  :M-r 1
	:K-p 300.0 :D-p 1000.0 :M-p 5)  ;;default :K-p 500.0 :D-p 1000.0 :M-p 5
  (send *ri* :start-impedance :larm)
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
  (send a :draw (mapcar #'(lambda (av) (elt av num)) list))
  ;; (setq b '())
  ;; (dotimes (i (length list))
  ;;   (push (aref (elt list i) num) b)
  ;;   )
  ;; (nreverse b)
  ;; (send a :draw b)
  )

(defun wrench-drawplot (&key (list *wrench-list*) (num 12))
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
	  ;;:cog-translation-axis :z
	  :root-link-virtual-joint-weight #f(0.1 0.1 0.1 0.1 1 10)
	  :addtional-weight-list (list (list (send *robot* :torso :waist-y :child-link) 1000.0)
	  			       (list (send *robot* :rleg :crotch-y :child-link) 100.0)
	  			       (list (send *robot* :lleg :crotch-y :child-link) 100.0)
	  			       (list (send *robot* :rarm :collar-y :child-link) 0.01)
				       )
	  ;; :collision-avoidance-link-pair (list (list (send *robot* :links :CHEST_LINK1) (send *robot* :links :LARM_LINK2)))
	  ;; :avoid-collision-joint-gain 1.0
	  ;; :avoid-collision-null-gain  200
	  )
    ;; (objects (list *robot* rarm-target))
    ;; (send *robot* :newcoords (make-coords))
    )
  )

(defun prioritized-ik (&key(rarm-trans-pos #f(0 0 0)) (rarm-trans-rpy #f(0 0 0))
			   (relbow-trans-pos #f(0 0 0)) (relbow-trans-rpy #f(0 0 0))
			   (larm-trans-pos #f(0 0 0)) (larm-trans-rpy #f(0 0 0))
			   (lelbow-trans-pos #f(0 0 0)) (lelbow-trans-rpy #f(0 0 0)) &rest args)
  (setq rarm-target (make-cascoords :coords (send *robot* :rarm :end-coords :copy-worldcoords)))
  (send rarm-target :translate rarm-trans-pos :world)
  (send rarm-target :transform (make-coords :pos #f(0 0 0) :rpy rarm-trans-rpy) :local)
  (setq relbow-target (make-cascoords :coords (send *robot* :rarm_link4_lk :copy-worldcoords)))
  (send relbow-target :translate relbow-trans-pos :world)
  (send relbow-target :transform (make-coords :pos #f(0 0 0) :rpy relbow-trans-rpy) :local)
  (setq larm-target (make-cascoords :coords (send *robot* :larm :end-coords :copy-worldcoords)))
  (send larm-target :translate larm-trans-pos :world)
  (send larm-target :transform (make-coords :pos #f(0 0 0) :rpy larm-trans-rpy) :local)
  (setq lelbow-target (make-cascoords :coords (send *robot* :larm_link4_lk :copy-worldcoords)))
  (send lelbow-target :translate lelbow-trans-pos :world)
  (send lelbow-target :transform (make-coords :pos #f(0 0 0) :rpy lelbow-trans-rpy) :local)
  (apply #'prioritized-inverse-kinematics
	   (list 
	    (list (instance move-target-task :init
			    (send *robot* :rarm :end-coords)
			    rarm-target
			    :thre 2.0
			    :rthre 0.1
			    :check nil)
		  (instance move-target-task :init
			    (send *robot* :larm :end-coords)
			    larm-target
			    :thre 2.0
			    :rthre 0.1
			    :check nil))
	    (list (instance move-target-task :init
			    (send *robot* :rarm_link4_lk)			  
			    relbow-target
			    :translation-axis t
			    ;; :rotation-axis nil
			    :check nil)
		  (instance move-target-task :init
			    (send *robot* :larm_link4_lk)			  
			    lelbow-target
			    :translation-axis t
			    ;; :rotation-axis nil
			    :check nil)))
	   (mapcar #'(lambda (j) (instance joint-variable :init j))
		   (append (send *robot* :rarm :joint-list) (send *robot* :larm :joint-list)))
	   :task0  (mapcar #'(lambda (j) (instance minmax-angle-task :init j))
			   (append (send *robot* :rarm :joint-list) (send *robot* :larm :joint-list)))
	   :min-loop 50
	   :debug-view nil
	   args)
  )

;;progn (reset-pose :run nil) (lleg-foot :run nil) (lower-waist :run nil) (posture :run nil)
;;(send *ri* :angle-vector (send *robot* :angle-vector) 10000)

;;;;send *ri* :start-auto-balancer
;;lleg-foot
;;lower-waist
;;posture
;;impendance
;;rotate :run t

;; progn (setq a biglist) nil
