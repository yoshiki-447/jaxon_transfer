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

(defun reset-pose (&key(run t))
  (send *robot* :reset-pose)
  (when run
    (send *ri* :angle-vector (send *robot* :angle-vector) 5000)
    )
  (send *robot* :fix-leg-to-coords (make-coords))
  )

(defun sim-irt ()
  (send *robot* :angle-vector (send *ri* :state :potentio-vector))
  (send *robot* :fix-leg-to-coords (make-coords))
  )

(defun lleg-foot (&key (run t))
  (let ((target-lleg-coords (send (send *robot* :lleg :end-coords :copy-worldcoords)
				  :translate (float-vector 0 150 0))
			    ;; :rotate (deg2rad 60) :z)))
			    ))
    (cond (run
	   (send *ri* :set-foot-steps
		 (list (make-coords :coords (send *robot* :rleg :end-coords :copy-worldcoords) :name :rleg)
		       (make-coords :coords target-lleg-coords :name :lleg)
		       )
		 )
	       (send *robot* :angle-vector (send *ri* :state :potentio-vector))
	       )
	  (t (send *robot* :lleg :inverse-kinematics target-lleg-coords))
	  )

    (send *robot* :fix-leg-to-coords (make-coords))
    ))

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

(defun move-waist (&key(dz) (tm))
       (send *robot* :fix-leg-to-coords (make-coords))
       (send *robot* :legs :move-end-pos (float-vector 0 0 dz))
       (send *robot* :move-centroid-on-foot :both '(:rleg :lleg))
       (send *ri* :angle-vector (send *robot* :angle-vector) tm)
       (send *ri* :wait-interpolation)
       ;; (send *robot* :newcoords (make-coords))
       (send *robot* :fix-leg-to-coords (make-coords))
       )

(defun upper-waist
    (&optional (dz -60) (tm 5000))
  (move-waist :dz dz :tm tm)
  )

(defun lower-waist
    (&optional (dz 60) (tm 5000))
  (move-waist :dz dz :tm tm)
  ;;(send *robot* :reset-pose)
  ;; (send *robot* :fix-leg-to-coords (make-coords))
  ;; (send *robot* :legs :move-end-pos (float-vector 0 0 dz))
  ;; (send *robot* :move-centroid-on-foot :both '(:rleg :lleg))
  ;; (send *ri* :angle-vector (send *robot* :angle-vector) tm)
  ;; (send *ri* :wait-interpolation)
  ;; (send *robot* :newcoords (make-coords))
  )

(defun start-pos (&key (run t))
  (send *robot* :larm :shoulder-p :joint-angle 20)

  (send *robot* :torso :waist-r :joint-angle 5)
  (send *robot* :torso :waist-p :joint-angle 20)

  (send *robot* :rarm :shoulder-p :joint-angle -40)
  (send *robot* :rarm :shoulder-r :joint-angle -30)
  (send *robot* :rarm :elbow-p :joint-angle -30)
  
  ;; (arm-solve-ik :larm-trans-pos #f(0 0 100))
  
  (send *robot* :larm :wrist-y :joint-angle -40)
  (send *robot* :larm :wrist-r :joint-angle 40)
  
  (send *robot* :rarm :wrist-r :joint-angle 40)

  (send *robot* :fix-leg-to-coords (make-coords))
  (send *robot* :move-centroid-on-foot :both '(:rleg :lleg))

  (when run
    (send *ri* :angle-vector (send *robot* :angle-vector) 5000)
    (send *ri* :wait-interpolation)
    )
  (full-solve-ik-locate :rarm-rpy (float-vector (deg2rad 45) (deg2rad 0) (deg2rad 0))
			:larm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad 90))
			:centroid 0.2)
  (when run
    (send *ri* :angle-vector (send *robot* :angle-vector) 5000)
    (send *ri* :wait-interpolation)
    )
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

(defun rarm-neck ()
  (full-solve-ik :rarm-trans-pos #f(300 0 200)
		 :rarm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad 0)) :centroid 0.2)
  (send *ri* :angle-vector (send *robot* :angle-vector) 5000)
  (send *ri* :wait-interpolation)
  (send *robot* :rarm :wrist-r :joint-angle 40)
  (send *ri* :angle-vector (send *robot* :angle-vector) 5000)
  (send *ri* :wait-interpolation)
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

;; (defun generate-circle-trajectory (center radius num-points)
(defun generate-circle-trajectory (&key (start-pos) (center2hand) (num-points))
  ;; centerは円の中心座標、radiusは半径、num-pointsは軌道上の点の数
  (let ((center (v- start-pos center2hand))
	(radius (norm center2hand))
	(theta-step (/ (/ pi 2) num-points))
        (trajectory '()))  ;; 結果を蓄積するリスト
    (dotimes (i (+ num-points 1))  ;; iが0から(num-points-1)までのループ
      (let ((theta (* i theta-step)))
	(setq target-circle (make-coords :pos (float-vector 0.0 (- 0.0 (* radius (cos theta))) (* radius (sin theta)))))
	(setq circle (v+ center (send target-circle :pos)))
        (cond ((> (length trajectory) 0)
	       (nconc trajectory (list circle))
	       )
	      (t
	       (push circle trajectory)
	       )
	      )
	)
      )
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
			   (send *robot* :link-list (send *robot* :larm :end-coords :parent)))
	  :translation-axis (list t t t t) 
	  ;; :rotation-axis (list t t nil nil)
	  ;; :rotation-axis (list t t t nil)
	  :rotation-axis (list t t t t)
	  :target-centroid-pos (midpoint centroid
	  				 (send *robot* :rleg :end-coords :worldpos)
	  				 (send *robot* :lleg :end-coords :worldpos))
	  :cog-translation-axis :z
	  )
    (objects (list *robot* rarm-target))
    ;; (send *robot* :newcoords (make-coords))
    )
  )

(defun follow-circle (&key (trajectory nil) (run nil) &allow-other-keys)
  ;;(setq r-angle 60)
  ;;(setq same-angle (/ r-angle (- (length trajectory) 1)))
  (setq start-value 0.2)
  (setq finish-value 0.8)
  (setq diff (/ (- finish-value start-value) (- (length trajectory) 1)))
  (setq l-start 90)
  (setq l-diff (/ l-start (- (length trajectory) 1)))
  
  (dolist (target trajectory)
    ;;(arm-solve-ik-locate :rarm-trans-pos target :rarm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad r-angle)))
    (full-solve-ik-locate :rarm-trans-pos target
			  :rarm-rpy (float-vector (deg2rad 45) (deg2rad 0) (deg2rad 0))
			  :larm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad l-start))
			  :centroid start-value)
    ;; (print "Enter to next")(read-line)
    (unix:usleep 200000)
    (when run
      (send *ri* :angle-vector (send *robot* :angle-vector) 1000)
      (send *ri* :wait-interpolation))
    (setq start-value (+ start-value diff))
    ;;(setq r-angle (- r-angle same-angle))
    (setq l-start (- l-start l-diff))
    )
  )


(defun wake (&rest args)
  ;; (arm-solve-ik :rarm-trans-pos #f(0 400 400)
  ;; 		:larm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad 0))
  ;; 		:rarm-rpy (float-vector (deg2rad 40) (deg2rad 0) (deg2rad 0)))  ;;起き上がらせる
  ;; (send *ri* :angle-vector (send *robot* :angle-vector) 5000)
  ;; (send *ri* :wait-interpolation)
  
  ;; (apply #'follow-circle (generate-circle-trajectory (v+ #f(0 400 0) (send *robot* :rarm :end-coords :worldpos)) 400 10) args)
  (apply #'follow-circle (append args (list :trajectory (generate-circle-trajectory :start-pos (send *robot* :rarm :end-coords :worldpos) :center2hand #f(0 -400 0) :num-points 10))))
  ;; (follow-circle :trajectory (generate-circle-trajectory :start-pos (send *robot* :rarm :end-coords :worldpos) :center2hand #f(0 -400 0) :num-points 10) :run nil)

  )

(defun wake2
    (&key(transpos #f(0 0 0)))

  ;; 動かす位置を右手のlink5に設定
  (setq move-target-coords (arrow))
  (send move-target-coords :newcoords (send *robot* :rarm_link5_lk :copy-worldcoords))

  ;; 目標位置の設定
  (setq target-coords (send
  		       (send (send *robot* :rarm_link5_lk :copy-worldcoords))
  		       :translate transpos :world))

  ;; assoc
  (send (send (send *robot* :rarm :end-coords) :parent) :assoc move-target-coords)
  
  (send *robot* :rarm :inverse-kinematics target-coords
  	:move-target move-target-coords)
  
  (setq rarm_link5 (send *robot* :rarm_link5_lk :copy-worldcoords))
  (format t "after rarm-link5: ~A~%" rarm_link5)

  ;; (objects (list *robot* rarm_link5 move-target-coords target-coords (send *robot* :rarm :end-coords :copy-worldcoords)))
  (objects (list *robot* target-coords move-target-coords))
  )

(defun impedance ()
  (send *ri* :set-impedance-controller-param :arms :K-r 200.0 :D-r 300.0  :M-r 1
	:K-p 500.0 :D-p 1000.0 :M-p 5)
  (send *ri* :start-impedance :larm)
  (send *ri* :set-ref-forces
	(list
	 (float-vector 0 0 0) ;;rleg
	 (float-vector 0 0 0) ;;lleg
	 (float-vector 0 -100  0) ;;rarm (x, y, z) 単位はニュートン
	 (float-vector 0 100   0) ;;larm
	 )
	3000)
  )

;;;;;;;;;;;;

(defun test-motion ()
  (send *robot* :reset-pose)
  (lleg-foot)
  (initial-pos)
  (bed-knee)
  (support)
  (lower-waist)
  (rarm-neck)
  (let ((traj (generate-circle-trajectory (v+ #f(0 200 0) (send *robot* :rarm :end-coords :worldpos)) 200 10)))
    (dolist (pos traj)
      (setq cube (make-cube 100 100 100))
      (send cube :translate pos)
      (full-solve-ik-locate :rarm-trans-pos pos)
      (objects (list *robot* cube))
      (send *ri* :angle-vector (send *robot* :angle-vector))
      (send *ri* :wait-interpolation)
      (read-line)
      )
    )
  )

