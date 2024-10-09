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

(defun reset-pose ()
  (send *robot* :reset-pose)
  (send *ri* :angle-vector (send *robot* :angle-vector) 5000)
  )

(defun lleg-foot ()
  (send *ri* :set-foot-steps
	(list (make-coords :coords (send *robot* :rleg :end-coords :copy-worldcoords) :name :rleg)
	      (make-coords :coords (send
				   ;; (send
				     (send *robot* :lleg :end-coords :copy-worldcoords)
				     :translate (float-vector 0 150 0))
				   ;; :rotate (deg2rad 60) :z)
			   :name :lleg)
	      )
	)
  (send *robot* :angle-vector (send *ri* :potentio-vector))
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

(defun upper-waist
  (&optional (dz -60) (tm 5000))
  ;(send *robot* :reset-pose)
  (send *robot* :fix-leg-to-coords (make-coords))
  (send *robot* :legs :move-end-pos (float-vector 0 0 dz))
  (send *robot* :move-centroid-on-foot :both '(:rleg :lleg))
  (send *ri* :angle-vector (send *robot* :angle-vector) tm)
  (send *ri* :wait-interpolation)
  (send *robot* :newcoords (make-coords))
  )

(defun lower-waist
  (&optional (dz 60) (tm 5000))
  ;(send *robot* :reset-pose)
  (send *robot* :fix-leg-to-coords (make-coords))
  (send *robot* :legs :move-end-pos (float-vector 0 0 dz))
  (send *robot* :move-centroid-on-foot :both '(:rleg :lleg))
  (send *ri* :angle-vector (send *robot* :angle-vector) tm)
  (send *ri* :wait-interpolation)
  (send *robot* :newcoords (make-coords))
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

(defun wake ()
  ;; (arm-solve-ik :rarm-trans-pos #f(0 400 400)
  ;; 		:larm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad 0))
  ;; 		:rarm-rpy (float-vector (deg2rad 40) (deg2rad 0) (deg2rad 0)))  ;;起き上がらせる
  ;; (send *ri* :angle-vector (send *robot* :angle-vector) 5000)
  ;; (send *ri* :wait-interpolation)
  
  (follow-circle (generate-circle-trajectory (v+ #f(0 200 0) (send *robot* :rarm :end-coords :worldpos)) 200 10))
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
  

;; (defun posture ()
;;   (lleg-foot)
;;   (initial-pos)
;;   (bed-knee)
;;   (support)
;;   (lower-waist)
;;   (rarm-neck)
;;   (wake)
;;   )

(defun generate-circle-trajectory (center radius num-points)
  ;; centerは円の中心座標、radiusは半径、num-pointsは軌道上の点の数
  (let ((theta-step (/ (/ pi 2) num-points))
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
	  :rotation-axis (list t t nil nil)
	  :target-centroid-pos (midpoint centroid
	  				 (send *robot* :rleg :end-coords :worldpos)
	  				 (send *robot* :lleg :end-coords :worldpos))
	  :cog-translation-axis :z
	  )
    (send *robot* :newcoords (make-coords))
    )
  )

(defun follow-circle (trajectory)
  ;;(setq r-angle 60)
  ;;(setq same-angle (/ r-angle (- (length trajectory) 1)))
  (setq start-value 0.2)
  (setq finish-value 0.8)
  (setq add (/ (- finish-value start-value) (- (length trajectory) 1)))
  
  (dolist (target trajectory)
    ;;(arm-solve-ik-locate :rarm-trans-pos target :rarm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad r-angle)))
    (full-solve-ik-locate :rarm-trans-pos target
			  :rarm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad 0))
			  :centroid start-value)
    (send *ri* :angle-vector (send *robot* :angle-vector) 1000)
    (send *ri* :wait-interpolation)
    (setq start-value (+ start-value add))
    ;;(setq r-angle (- r-angle same-angle))
    )
  )

(defun full-solve-ik
    (&key(rarm-trans-pos #f(0 0 0))
	 (larm-trans-pos #f(0 0 0))
	 (rarm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad 0)))
	 (larm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad 0)))
	 (centroid 0.5))
  (let* ((rleg-coords (send *robot* :rleg :end-coords :copy-worldcoords))
	 (lleg-coords (send *robot* :lleg :end-coords :copy-worldcoords))
	 (rarm-end-coords (send *robot* :rarm :end-coords :copy-worldcoords))
	 (larm-end-coords (send *robot* :larm :end-coords :copy-worldcoords))
	 (rarm-translation-coords (send rarm-end-coords :translate rarm-trans-pos :world) :copy-worldcoords)
	 (larm-translation-coords (send larm-end-coords :translate larm-trans-pos :world) :copy-worldcoords)
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
	  :rotation-axis (list t t nil nil)
	  :target-centroid-pos (midpoint centroid
	  				 (send *robot* :rleg :end-coords :worldpos)
	  				 (send *robot* :lleg :end-coords :worldpos))
	  :cog-translation-axis :z
	  )
    (send *robot* :newcoords (make-coords))
    )
  )

;;;;;;;;;;;;

(defun test-motion ()
  (setq *move-target-coords* (arrow))
  (send *move-target-coords* :newcoords (send *robot* :rarm_link5_lk :copy-worldcoords) )

  (setq *target-coords* (send (send *robot* :rarm_link5_lk :copy-worldcoords) :translate #f(0 0 0) :world))

  (send (send (send *robot* :rarm_link5_lk) :parent) :assoc *move-target-coords*)

  (objects (list *robot* *move-target-coords* *target-coords*))

  (unix::usleep (* 2 1000000))

  ;; (send *robot* :inverse-kinematics (list target-coords)
  ;; 	:move-target (list move-target-coords)
  ;; 	:link-list (list (send (send *robot* :rarm :end-coords) :parent))
  ;; 	)
  (send *robot* :rarm :inverse-kinematics *target-coords*
  	:move-target *move-target-coords*
  	)

  (objects (list *robot* *move-target-coords* *target-coords*))
  )
