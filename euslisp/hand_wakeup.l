(require "models/arrow-object.l")
(load "irteus/demo/sample-robot-model.l")

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
  (send *robot* :reset-pose)
  (send *robot* :fix-leg-to-coords (make-coords))
  (let ((target-lleg-coords1 (send
			      (send *robot* :lleg :end-coords :copy-worldcoords)
			      :translate (float-vector 0 100 0))
			     ;; :rotate (deg2rad 60) :z)))
			     )
	(target-lleg-coords2 (send
			      (send
			       (send *robot* :lleg :end-coords :copy-worldcoords)
			       :translate (float-vector -200 100 0))
			      :rotate (deg2rad 30) :z)
			     )
	(target-rleg-coords (send
			     (send *robot* :rleg :end-coords :copy-worldcoords)
			     :translate (float-vector 0 -150 0))
			    ;; :rotate (deg2rad 60) :z)))
			    ))
    (cond (run
	   (send *ri* :set-foot-steps
		 (list (make-coords :coords (send *robot* :rleg :end-coords :copy-worldcoords) :name :rleg)
		       (make-coords :coords target-lleg-coords1 :name :lleg)
		       (make-coords :coords target-rleg-coords :name :rleg)
		       (make-coords :coords target-lleg-coords2 :name :lleg)
		       )
		 )
	   ;; (send *ri* :wait-interpolation)
	   ;; (unix::sleep 7)
	   ;; (send *robot* :angle-vector (send *ri* :state :potentio-vector))
	   (send *robot* :lleg :inverse-kinematics target-lleg-coords1)
	   (send *robot* :lleg :inverse-kinematics target-lleg-coords2)
	   (send *robot* :rleg :inverse-kinematics target-rleg-coords)
	   )
	  (t
	   (send *robot* :lleg :inverse-kinematics target-lleg-coords1)
	   (send *robot* :lleg :inverse-kinematics target-lleg-coords2)
	   (send *robot* :rleg :inverse-kinematics target-rleg-coords)
	   )
	  )

    ;; (send *robot* :fix-leg-to-coords (make-coords))
    (send *robot* :move-centroid-on-foot :both '(:rleg :lleg))
    )
  )

(defun fix-coords ()  ;;after lower-waist
  (setq a (make-coords))
  (send a :translate (float-vector (aref (send (send *robot* :foot-midcoords) :pos) 0) (aref (send (send *robot* :foot-midcoords) :pos) 1) 0))  ;;lower-waist z しても０にするため
  (send a :rotate (deg2rad 15) :z)  ;;lleg-foot で３０度回転
  (send *robot* :fix-leg-to-coords a)
  (format t "a: ~A~%" a)
  )

(defun move-waist (&key(dz) (tm) (run t))
       ;; (send *robot* :fix-leg-to-coords (make-coords))
       (send *robot* :legs :move-end-pos (float-vector 0 0 dz))
       (send *robot* :move-centroid-on-foot :both '(:rleg :lleg))
       (when run
	 (send *ri* :angle-vector (send *robot* :angle-vector) tm)
	 (send *ri* :wait-interpolation)
	 ;; (send *robot* :angle-vector (send *ri* :state :potentio-vector))
	 )
       ;; (send *robot* :newcoords (make-coords))
       ;; (send *robot* :fix-leg-to-coords (make-coords))
       (fix-coords)
       )

(defun upper-waist
    (&key (dz -60) (tm 5000) (run t))
  (move-waist :dz dz :tm tm :run run)
  )

(defun lower-waist
    (&key (dz 100) (tm 5000) (run t))  ;;100
  (move-waist :dz dz :tm tm :run run) 
  )

(defun link-end (&key(link-name :rarm_link0_lk))  ;;rarm-end-coordsとのxy距離
  (setq collar (float-vector (aref (send *robot* link-name :worldpos) 0) (aref (send *robot* link-name :worldpos) 1) (aref (send *robot* :rarm :end-coords :worldpos) 2)))
  (setq collar2end (v- (send *robot* :rarm :end-coords :worldpos) collar))
  (norm collar2end)
  )

(defun manequine (&key (wake nil))
  (setq *human* (instance sample-robot :init))
  (send *human* :move-to (make-coords :pos #f(600 -200 900) :rpy (float-vector (deg2rad 180) (deg2rad 0) (deg2rad -90))))
  (send *human* :rleg :crotch-p :joint-angle -30)
  (send *human* :rleg :knee-p :joint-angle 30)
  (send *human* :lleg :crotch-p :joint-angle -30)
  (send *human* :lleg :knee-p :joint-angle 30)
  (send *human* :rarm :elbow-p :joint-angle -120)
  (send *human* :rarm :shoulder-p :joint-angle -20)
  (send *human* :rarm :shoulder-y :joint-angle -60)
  (send *human* :larm :elbow-p :joint-angle -120)
  (send *human* :larm :shoulder-p :joint-angle -20)
  (send *human* :larm :shoulder-y :joint-angle -60)
  
  (when wake
    (send *human* :rleg :crotch-p :joint-angle -60)
    (send *human* :rleg :knee-p :joint-angle 60)
    (send *human* :lleg :crotch-p :joint-angle -60)
    (send *human* :lleg :knee-p :joint-angle 60)
    (send *human* :move-to (make-coords :rpy (float-vector 0 0 (deg2rad 90))))
    (send *human* :locate #f(600 350 900) :world)
    )
  )

(defun posture (&key (run t))
  (send *robot* :torso :waist-p :joint-angle 20)
  (full-solve-ik-locate :rarm-trans-pos #f(600 -500 750) 
  			:larm-trans-pos #f(600 250 1050)
  			:rarm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad -30))
  			:larm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad 90))
  			:centroid 0.2
  			:chest t)  ;;success mocap1 mocap3
  ;; (send *robot* :torso :waist-p :joint-angle 20)
  ;; (full-solve-ik-locate :rarm-trans-pos #f(600 -500 750) 
  ;; 			:larm-trans-pos #f(600 250 1000)
  ;; 			:rarm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad 90))
  ;; 			:larm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad 90))
  ;; 			:centroid 0.2
  ;; 			:chest t)  ;;neck mocap2
  (when run
    (send *ri* :change-cop 0.0 -0.15 0.0 -0.15 5.0)
    (send *ri* :wait-interpolation)
    (send *ri* :angle-vector (send *robot* :angle-vector) 5000)
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

(defvar *r-theta-list* '())
(defvar *l-theta-list* '())
(setq init-num 1000)
(defvar *wrench-list* '())
(defun generate-circle-trajectory (&key (r-start-pos) (r-radius) (r-angle) (r-finish-angle) (r-phi) (l-start-pos) (l-radius) (l-angle) (l-finish-angle) (num-points) (mass))
  (setq *wrench-list* '())
  (setq cube-list '())
  (setq grav 9.8)
  (setq force_fit 30.0) ;;?
  (setq force_adjust 50.0)
  ;; (setq b 0.9)
  ;; (setq r-daen (* r-radius (/ b (sqrt(+ (expt (* b (cos r-angle)) 2) (expt (sin r-angle) 2))))))
  ;; (print r-daen)
  (let* ((r-center2hand (float-vector 0.0 (- 0.0 (* r-radius (cos r-angle))) (* 0.9 r-radius (sin r-angle))))  ;;mocap 1,2 0.9 mocap3 z1.0
	 (r-center (v- r-start-pos r-center2hand))
	 (l-center2hand (float-vector 0.0 (* 1.2 l-radius (sin l-angle)) (* l-radius (cos l-angle))))
	 (l-center (v- l-start-pos l-center2hand))
	 (r-trajectory '())
	 (l-trajectory '()))
    ;; (format t "r-center:~A~%" r-center)
    ;; (setq c1 (make-cube 100 100 100))
    ;; (setq c2 (make-cube 100 100 100))
    ;; (send c1 :locate center :world)
    ;;
  ;; (setq b '())
    ;;
    (setq *r-theta-list* '())
    (setq *l-theta-list* '())
    (dotimes (i (+ num-points 1))
      ;; (print "Enter to next")(read-line)
      ;; (let ((theta (* i theta-step)))
      ;; (setq theta (+ theta angle))
      ;; (let ((theta (theta-cos :a (/ (- angle finish-angle) 2.0)
      ;; 			      :b (/ pi num-points)
      ;; 			      :index i
      ;; 			      :c (+ (/ (- finish-angle angle) 2.0) angle)
      ;; 			      )))
      (let ((r-theta (minimum-leap :x1 0.0
			      :y1 r-angle
			      :x2 (float num-points)
			      :y2 r-finish-angle
			      :index (float i)
			      ))
	    (r-phi-theta (minimum-leap :x1 0.0
				       :y1 0.0
				       :x2 (float num-points)
				       :y2 r-phi
				       :index (float i)
				       ))
	    (l-theta (minimum-leap :x1 0.0
			      :y1 l-angle
			      :x2 (float num-points)
			      :y2 l-finish-angle
			      :index (float i)
			      )))
	;;
;;	(push theta b)
	;;
	
	(push (rad2deg r-theta) *r-theta-list*)
	(push (rad2deg l-theta) *l-theta-list*)
	
	;; (setq r-target-circle (float-vector 0.0 (- 0.0 (* r-radius (cos r-theta))) (* 0.9 (* r-radius (sin r-theta))))) ;;waki1.1
	(setq r-target-circle (float-vector (- (* 0.5 r-radius (sin r-theta) (sin r-phi-theta))) (- (* r-radius (cos r-theta)))
					    (* 0.9 r-radius (sin r-theta) (cos r-phi-theta))))  ;;mocap 1,2 0.9 mocap3z 1.0
	(setq r-circle (v+ r-center r-target-circle))  ;;rarm
	(setq l-target-circle (float-vector 0.0 (* 1.2 (* l-radius (sin l-theta))) (* 1.0 (* l-radius (cos l-theta))))) ;;waki1.4
	(setq l-circle (v+ l-center l-target-circle))  ;;larm

	(setq r_force (/ (* (* mass grav) (cos r-theta)) 2.0))
	;;(setq r_moment (* r_force (/ r-radius 1000)))
	;; (setq r_fx (theta-3 :x1 0.0 :y1 0.0 :x2 (float num-points) :y2 -100 :index (float i)))
	(cond ((> i (/ (* num-points 2) 3))
	       (setq r_fx -30.0)
	       )
	      (t
	       (setq r_fx (minimum-leap :x1 0.0 :y1 0.0 :x2 (float (/ (* num-points 2) 3)) :y2 -30.0 :index (float i)))
	       )
	      )
	;; (cond ((and (> i 2000) (>= r_fy (- force_adjust)))
	;;        (setq r_fy force_adjust)
	;;        )
	;;       (t
	;;        (setq r_fy (+ (* r_force (sin r-theta)) (* force_fit (cos r-theta))))
	;;        )
	;;       )
	(setq r_fy (+ (* r_force (sin r-theta)) (* force_fit (cos r-theta))))
	(setq r_fz (- (* r_force (cos r-theta)) (* force_fit (sin r-theta))))
	(setq r_fx (- r_fx))
	(setq r_fy (- r_fy))
	(setq r_fz (- r_fz))  ;;ロボットが受ける力
	;; (incf r_corr 0.5)
	;; (when (< r_corr r_fy)
	;;   (setq r_fy r_corr)
	;;   )
	;; (when (< r_corr r_fz)
	;;   (setq r_fz r_corr)
	;;   )
	;; (setq r_Mx (/ (- (* (aref r-circle 1) r_fz) (* (aref r-circle 2) r_fy)) 1000))
	;; (setq r_Mx (/ (- (* (aref r-target-circle 1) r_fz) (* (aref r-target-circle 2) r_fy)) 1000))
	(setq r_Mx 0.0)
	(setq r_My 0.0)
	(setq r_Mz 0.0)
	(setq r_wr (float-vector r_fx r_fy r_fz r_Mx r_My r_Mz))
	;; (setq r_wr (transform (transform-matrix :rarm t) r_wr))
	(setq l_force 25.0)
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
    ;;   ;; (setq rx_corr (minimum-leap :x1 0.0 :y1 0.0 :x2 (float init-num) :y2 (elt (car *wrench-list*) 12) :index (float i)))
    ;;   (setq ry_corr (minimum-leap :x1 0.0 :y1 0.0 :x2 (float init-num) :y2 (elt (car *wrench-list*) 13) :index (float i)))
    ;;   (setq rz_corr (minimum-leap :x1 0.0 :y1 0.0 :x2 (float init-num) :y2 (elt (car *wrench-list*) 14) :index (float i)))
    ;;   (setq rmx_corr (minimum-leap :x1 0.0 :y1 0.0 :x2 (float init-num) :y2 (elt (car *wrench-list*) 15) :index (float i)))
    ;;   (setq lz_corr (minimum-leap :x1 0.0 :y1 0.0 :x2 (float init-num) :y2 (elt (car *wrench-list*) 20) :index (float i)))
    ;;   (setq init-wr (float-vector 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ry_corr rz_corr rmx_corr 0.0 0.0 0.0 0.0 lz_corr 0.0 0.0 0.0))
    ;;   (push init-wr init-list-wr)
    ;;   )
    ;; (nreverse init-list-wr)
    ;; (setq *wrench-list* (append init-list-wr *wrench-list*)) ;;this
    ;;
    ;; (setq a (gnuplot))
    ;; (nreverse b)
    ;; (print b)
    ;; (send a :draw b)
    ;;

    ;; sample human
    ;; (setq coords (arrow))
    ;; (push coords cube-list)
    ;; (push *robot* cube-list)
    ;; (push *human* cube-list)
    ;; (objects cube-list)
    
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
	 (centroid 0.5)
	 (axis :z)
	 (chest nil))
  (let* ((rleg-coords (send *robot* :rleg :end-coords :copy-worldcoords))
	 (lleg-coords (send *robot* :lleg :end-coords :copy-worldcoords))
	 (rarm-end-coords (send *robot* :rarm :end-coords :copy-worldcoords))
	 (larm-end-coords (send *robot* :larm :end-coords :copy-worldcoords))
	 (rarm-translation-coords (send rarm-end-coords :locate rarm-trans-pos :world) :copy-worldcoords)
	 (larm-translation-coords (send larm-end-coords :locate larm-trans-pos :world) :copy-worldcoords)
	 (rarm-target (make-coords :pos (send rarm-translation-coords :pos)
				   :rpy rarm-rpy))
	 (larm-target (make-coords :pos (send larm-translation-coords :pos)
				   :rpy larm-rpy))
	 (link-list (list (send *robot* :link-list (send *robot* :rleg :end-coords :parent))
			  (send *robot* :link-list (send *robot* :lleg :end-coords :parent))))
	 (remove-link #'(lambda (target-link link-list)
			  (remove target-link link-list
				  :test #'(lambda (val1 val2) (equal (send val1 :name) (send val2 :name))))))
	 )
    (cond (chest
	   (nconc link-list (list (send *robot* :link-list (send *robot* :rarm :end-coords :parent) (send *robot* :chest_link2_lk)))) 
	   (nconc link-list (list (send *robot* :link-list (send *robot* :larm :end-coords :parent) (send *robot* :chest_link2_lk))))
	   )
	  (t	   
	   ;; (nconc link-list (list (send *robot* :link-list (send *robot* :rarm :end-coords :parent))))
	   ;; (nconc link-list (list (send *robot* :link-list (send *robot* :larm :end-coords :parent))))
	   (nconc link-list (list (funcall remove-link
					   (send *robot* :rarm_link0_lk)
					   (send *robot* :link-list (send *robot* :rarm :end-coords :parent)))))
	   (nconc link-list (list (funcall remove-link
					   (send *robot* :larm_link0_lk)
					   (send *robot* :link-list (send *robot* :larm :end-coords :parent)))))
	   )
	  )
    (send *robot* :fullbody-inverse-kinematics
	  (list rleg-coords lleg-coords rarm-target larm-target)
	  :move-target (list (send *robot* :rleg :end-coords)
			     (send *robot* :lleg :end-coords)
			     (send *robot* :rarm :end-coords)
			     (send *robot* :larm :end-coords))
	  :link-list link-list
	  :translation-axis (list t t t t) 
	  ;; :rotation-axis (list t t nil nil)
	  ;; :rotation-axis (list t t t nil)
	  :rotation-axis (list t t t t)
	  :target-centroid-pos (midpoint centroid
	  				 (send *robot* :rleg :end-coords :worldpos)
	  				 (send *robot* :lleg :end-coords :worldpos))
	  :cog-translation-axis axis
	  :addtional-weight-list (list (list (send *robot* :rarm :collar-y :child-link) 0.01))
	  :thre (list 1.0 1.0 0.1 0.1)
	  :rthre (list (deg2rad 1.0) (deg2rad 1.0) 0.01 0.01)
	  )
    ;;(objects (list *robot* rarm-target))
    ;; (send *robot* :newcoords (make-coords))
    )
  )

(defvar *angle-list '())
(defvar *time-list* '())
(defun follow-circle (&key (r_tra nil) (l_tra nil) (run nil) &allow-other-keys)
  (setq *angle-list* '())
  (setq *time-list* '())
  (send *ri* :set-interpolation-mode :linear)
  (setq init-r-roll (rad2deg (elt (car (send (send *robot* :rarm :end-coords :copy-worldcoords) :rpy-angle)) 2)))
  (setq init-l-roll (rad2deg (elt (car (send (send *robot* :larm :end-coords :copy-worldcoords) :rpy-angle)) 2)))
  (dotimes (i (length r_tra))
    (setq r_target (elt r_tra i))
    (setq l_target (elt l_tra i))
    (setq start-value (minimum-leap :x1 0.0
				    :y1 0.2
				    :x2 (- (length r_tra) 1)
				    :y2 0.8
				    :index (float i)))
    (setq r-start (minimum-leap :x1 0.0
			 :y1 init-r-roll
			 :x2 (- (length r_tra) 1)
			 :y2 -90.0  ;;mocap1,3 -90.0 mocap2 0.0
			 :index (float i)))
    (setq l-start (minimum-leap :x1 0.0
			 :y1 init-l-roll
			 :x2 (- (length r_tra) 1)
			 :y2 0.0
			 :index (float i)))
    (setq r-yaw (minimum-leap :x1 0.0
			 :y1 0.0
			 :x2 (- (length r_tra) 1)
			 :y2 20.0  ;;mocap1 20 mocap2,3 0
			 :index (float i)))
    (full-solve-ik-locate :rarm-trans-pos r_target
    			  :larm-trans-pos l_target
    			  :rarm-rpy (float-vector (deg2rad r-yaw) (deg2rad 0) (deg2rad r-start))
    			  :larm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad l-start))
    			  :centroid start-value
    			  :axis nil
			  :chest nil)
    ;; (unix:usleep 1000)
    (send *irtviewer* :draw-objects)
    (x::window-main-one)
    
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

  ;;(nreverse cop-list)
  
  ;; (dotimes (i init-num)
  ;;   ;; (push init-angle *angle-list*)
  ;;   (push (car *angle-list*) *angle-list*)
  ;;   (push 2.0 *time-list*)
  ;;   ) ;;this

  (setq start-num 100)
  
  (when run
    ;; (send *ri* :start-log)
    ;; (send *ri* :angle-vector (car *angle-list*))
    ;; (send *ri* :wait-interpolation) ;;this
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
    ;;(send *ri* :angle-vector-sequence-full *angle-list* *time-list* :wrench *wrench-list*) ;;this
    (send *ri* :angle-vector-sequence-full (nthcdr start-num *angle-list*) (nthcdr start-num *time-list*)
  	  :wrench (nthcdr start-num *wrench-list*))
    (send *ri* :change-cop 0.0 0.1 0.0 0.1 5.0)  ;; default rleg_x 0.0 rleg_y 0.02 lleg_x 0.0 lleg_y -0.02 t 2.0 ;;time10,5s
    
    ;; (send *ri* :set-auto-stabilizer-param :default-zmp-offsets (list #f(0 0.02) #f(0 -0.02)))
    ;; (unix:sleep 0.5)
    ;; (send *ri* :set-auto-stabilizer-param :default-zmp-offsets (list #f(0 0.02) #f(0 -0.02)))
    ;; (send *ri* :wait-interpolation)
    )
  )


(defun wake (&rest args)  
  ;; (apply #'follow-circle (generate-circle-trajectory (v+ #f(0 400 0) (send *robot* :rarm :end-coords :worldpos)) 400 10) args)
  (multiple-value-setq (r-tra l-tra) (generate-circle-trajectory :r-start-pos (send *robot* :rarm :end-coords :worldpos)
								 :r-radius 600.0 :r-angle (deg2rad 10) :r-finish-angle (deg2rad 100)
								 :r-phi (deg2rad 10) ;;mocap1,2 10 mocap3 0
								 :l-start-pos (send *robot* :larm :end-coords :worldpos)
								 :l-radius 200.0 :l-angle (deg2rad 0) :l-finish-angle (deg2rad 90)
								 :num-points 2500 :mass 25.0))  ;;waki r-radius300 ;;neck450 ;;num-points 10s 5000
  (apply #'follow-circle (append args (list :r_tra r-tra :l_tra l-tra)))  ;;larm200or275
  )

(defun impedance ()
  (send *ri* :set-impedance-controller-param :larm :K-r 200.0 :D-r 300.0  :M-r 1
	:K-p 300.0 :D-p 1000.0 :M-p 5)  ;;default :K-p 500.0 :D-p 1000.0 :M-p 5
  (send *ri* :set-impedance-controller-param :rarm :K-r 200.0 :D-r 300.0  :M-r 1
	:K-p 300.0 :D-p 1000.0 :M-p 5 :force-gain (list 0 1.0 0)) ;;mocap2 waki (list 0 0 1.0) mocap1,3 (list 0 1.0 0)
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

;;progn (reset-pose :run nil) (lleg-foot :run nil) (lower-waist :run nil) (posture :run nil)

;;progn (reset-pose :run nil) (lleg-foot :run nil) (lower-waist :run nil)
;;irt-ri


;;send *ri* :start-auto-balancer
;;lleg-foot
;;lower-waist
;;start-pos
;;impendance
;;下ろす
;;send *ri* :set-auto-stabilizer-param :is-emergency-step-mode nil
;;send *ri* :start-st
;;wake :run t
