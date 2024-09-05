(ros::roseus "recognition")
(ros::roseus-add-msgs "jsk_recognition_msgs")

;; (ros::rate 20)
;; (:callback (msg)
;; 	   (setq 
;; (ros::subscribe "/edgetpu_human_pose_estimator/output/poses" jsk_recognition_msgs::PeoplePoseArray #'send self:callback 1)

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
    (setq new-angle-vector
	  (send *robot* :inverse-kinematics (list rarm-target larm-target)
		:move-target (list (send *robot* :rarm :end-coords)
				   (send *robot* :larm :end-coords))
		:link-list (list (send *robot* :link-list (send *robot* :rarm :end-coords :parent))
				 (send *robot* :link-list (send *robot* :larm :end-coords :parent)))
		:translation-axis (list t t)
		:rotation-axis (list t t)
	        :debug-view nil
		;; :target-centroid-pos (midpoint 0.5
		;; 				 (send *robot* :rleg :end-coords :worldpos)
		;; 				 (send *robot* :rleg :end-coords :worldpos))
		;; :cog-translation-axis :z
		)
	  )
    new-angle-vector ;; 返り値
    )
  )

;;input: カメラ座標系の座標
;;output: rootmethodlink座標系の座標
(defun world-to-target (target)
  (let target-coords
    ;; ikを解く前にrobotに実機の姿勢を反映
    ;; (send *robot* :angle-vector (send *ri* :state :potentio-vector))
    ;; (send robot :fix-leg-to-coords (make-coords))
    (setq target-coords
      (send (send *robot* :rs_l515_depth_optical_frame :copy-worldcoords) :transform target))
    target-coords)
  )

(defun recognition_end (msg string1 string2)
   (when (> (length (send msg :skeletons)) 0) ;; 一人以上認識したら処理をする
      (setq pose-msg (car (send msg :skeletons)))
      (setq limb-msg (send pose-msg :bone_names))
      (let ((count 0) (success nil))
        (dolist (limb-name limb-msg)
	  (when (string-equal limb-name string1)
	    (setq shoulder-index count)
	    (setq success t)
	    )
	  (incf count)
	  )
	(when success
	  ;;(format t "~A index: ~A~%" string2 shoulder-index)
	  (let* ((coords1 (make-coords :pos (ros::tf-point->pos (send (elt (send pose-msg :bones) shoulder-index) :end_point))
				       :rpy (float-vector 0 0 0)))
		 (coords2 (world-to-target coords1)))
	    (setq coords coords2)
	    ;;(print coords)
	    (setq pos (send coords :pos))
	    ;;(print (send coords1 :pos))
	    ;;(print pos)
	    pos
	    )
	  )
	)
      )
   )

(defun recognition_start (msg string1 string2)
   (when (> (length (send msg :skeletons)) 0) ;; 一人以上認識したら処理をする
      (setq pose-msg (car (send msg :skeletons)))
      (setq limb-msg (send pose-msg :bone_names))
      (let ((count 0) (success nil))
        (dolist (limb-name limb-msg)
	  (when (string-equal limb-name string1)
	    (setq shoulder-index count)
	    (setq success t)
	    )
	  (incf count)
	  )
	(when success
	  ;;(format t "~A index: ~A~%" string2 shoulder-index)
	  (let* ((coords1 (make-coords :pos (ros::tf-point->pos (send (elt (send pose-msg :bones) shoulder-index) :start_point))
				       :rpy (float-vector 0 0 0)))
		 (coords2 (world-to-target coords1)))
	    (setq coords coords2)
	    ;;(print coords)
	    (setq pos (send coords :pos))
	    ;;(print (send coords1 :pos))
	    ;;(print pos)
	    pos
	    )
	  )
	)
      )
   )

(defun hold_point (shoulder-pos elbow-pos hip-pos &key(m-shoulder 2) (m-elbow 2) (m-hip 1))
  (setq m (+ m-shoulder m-elbow m-hip))
  ;; (print m)
  (setq v_shoulder (scale m-shoulder shoulder-pos))
  (setq v_elbow (scale m-elbow elbow-pos))
  (setq v_hip (scale m-hip hip-pos))
  (print v_shoulder)
  (print v_elbow)
  (print v_hip)
  ;; (print (/ 1.0 m))
  ;; (print (v+ v_shoulder v_elbow))
  (setq vector1 (v+ v_shoulder v_elbow))
  (setq hold_point (scale (/ 1.0 m) (v+ vector1 v_hip)))
  ;; (setq vector (v+ (v+ v_shoulder v_elbow) v_hip))
  ;; ;;(setq hold_point (scale (/ 1.0 m) (v+ v_shoulder v_elbow v_hip)))
  hold_point
  )
;; (setq *r-shoulder-list* '())
;; (setq *l-elbow-list* '())

(defclass Subscriber
  :super propertied-object
  :slots (limb-msg pose-msg shoulder-index r-shoulder-coords r-shoulder-pos)
  )

(defmethod Subscriber
  (:init () (setq limb-msg nil
		  pose-msg nil
		  shoulder-index nil
		  r-shoulder-coords nil
		  r-shoulder-pos nil)
	 self)
  
  (:ros-init ()
    ;;(ros::roseus "sub")
    (ros::subscribe "/skeleton_with_depth/output/skeletons" ;; subscribeするtopicの名前を指定
            jsk_recognition_msgs::HumanSkeletonArray ;; msgの型
            #'send self :cb-function ;; call-backメソッドを指定
            )
   )

  
  (:cb-function (msg)
    (setq l-shoulder-pos nil)
    (setq r-shoulder-pos nil)
    (setq l-elbow-pos nil)
    (setq r-elbow-pos nil)
    (setq l-hip-pos nil)
    (setq r-hip-pos nil)
    (setq l-shoulder-pos (recognition_start msg "left shoulder->right shoulder" "left-shoulder"))
    (setq r-shoulder-pos (recognition_end msg "left shoulder->right shoulder" "right-shoulder"))
    (setq l-elbow-pos (recognition_end msg "left shoulder->left elbow" "left-elbow"))		
    (setq r-elbow-pos (recognition_end msg "right shoulder->right elbow" "right-elbow"))
    (setq l-hip-pos (recognition_start msg "left hip->right hip" "left-hip"))
    (setq r-hip-pos (recognition_end msg "left hip->right hip" "right-hip"))
    
    ;; (push (recognition msg "left shoulder->right shoulder" "right-shoulder") *r-shoulder-list*)
    ;; (push (recognition msg "right shoulder->left elbow" "left-elbow") *l-elbow-list*)
    ;; (print *r-shoulder-list*)
    ;; (print *l-elbow-list*)
    ;; (when (> (length (send msg :skeletons)) 0) ;; 一人以上認識したら処理をする
    ;;   (setq pose-msg (car (send msg :skeletons)))
    ;;   (setq limb-msg (send pose-msg :bone_names))
    ;;   (let ((count 0) (success nil))
    ;;     (dolist (limb-name limb-msg)
    ;; 	  (when (string-equal limb-name "left shoulder->right shoulder")
    ;; 	    (setq shoulder-index count)
    ;; 	    (setq success t)
    ;; 	    )
    ;; 	  (incf count)
    ;; 	  )
    ;; 	(when success
    ;; 	  (format t "right-shoulder index: ~A~%" shoulder-index)
    ;; 	  (let* ((coords1 (make-coords :pos (ros::tf-point->pos (send (elt (send pose-msg :bones) shoulder-index) :end_point))
    ;; 				       :rpy (float-vector 0 0 0)))
    ;; 		 (coords2 (world-to-target coords1)))
    ;; 	    (setq r-shoulder-coords coords2)
    ;; 	    ;;(print r-shoulder-coords)
    ;; 	    (print (send r-shoulder-coords :pos))
    ;; 	    (setq r-shoulder-pos (send r-shoulder-coords :pos))
    ;; 	    )
    ;; 	  )
    ;; 	)
    ;;   )
    
    )

  (:ros-motion ()
    (while (ros::ok)
      (ros::spin-once)
      ;; ここにsubscribeした結果を元にした動作メソッドとかを入れたりする
      ;;抱える
      (when (and r-shoulder-pos r-elbow-pos r-hip-pos)
	;;(format t "r-shoulder: ~A~%r-elbow: ~A~%r-hip: ~A~%" r-shoulder-pos r-elbow-pos r-hip-pos)
	(setq r_hold_point (hold_point r-shoulder-pos r-elbow-pos r-hip-pos))
	(format t "r_hold_point: ~A~%" r_hold_point)
	;;(format t "norm: ~A~%" (norm (v- r-shoulder-pos r-elbow-pos)))
	;; (format t "norm: ~A~%" (norm r-shoulder-pos))
	;; (format t "norm: ~A~%" (norm l-elbow-pos))
	
	;; (let* ((limit 1000)
	;;        (dist-check-r-hold-point (< (norm r_hold_point) limit)))
	;;   (cond (dist-check-r-hold-point
	;; 	 ;;(arm-solve-ik-locate :larm-trans-pos r_hold_point)
	;; 	 (print "success")
	;; 	 )
	;; 	(t
	;; 	 (format t "ik failed~%")
	;; 	 )
	;; 	)
	;;   )
	)
      (when (and l-shoulder-pos l-elbow-pos l-hip-pos)
	;;(format t "l-shoulder: ~A~%l-elbow: ~A~%l-hip: ~A~%" l-shoulder-pos l-elbow-pos l-hip-pos)
	(setq l_hold_point (hold_point l-shoulder-pos l-elbow-pos l-hip-pos))
	(format t "l_hold_point: ~A~%" l_hold_point)
	
	;; (let* ((limit 1000)
	;;        (dist-check-l-hold-point (< (norm l_hold_point) limit)))
	;;   (cond (dist-check-l-hold-point
	;; 	 ;;(arm-solve-ik-locate :rarm-trans-pos l_hold_point)
	;; 	 (print "success")
	;; 	 )
	;; 	(t
	;; 	 (format t "ik failed~%")
	;; 	 )
	;; 	)
	;;   )
	)
      (ros::sleep)
      )
    )

  ;; (:ros-motion ()
  ;;   (while (ros::ok)
  ;;     (ros::spin-once)
  ;;     ;; ここにsubscribeした結果を元にした動作メソッドとかを入れたりする
  ;;     (let* ((limit 400)
  ;; 	     (dist-check (< (norm r-shoulder-pos) limit)))
  ;; 	;; (let ((ik-result (arm-solve-ik-locate :larm-trans-pos r-shoulder-pos)))
  ;; 	(when ik-result ;;r-shoulder-coords    
  ;; 	  ;;(print "success")
  ;; 	  (arm-solve-ik-locate :larm-trans-pos r-shoulder-pos)
  ;; 	  )
  ;; 	)
  ;;     ;; (when *r-shoulder-list*
  ;;     ;; 	(arm-solve-ik-locate :larm-trans-pos (send (car *r-shoulder-list*) :pos))
  ;;     ;; 	)
  ;;     ;;  (when *l-elbow-list*
  ;; 	;; 	(arm-solve-ik-locate :rarm-trans-pos (send (car *l-elbow-list*) :pos))
  ;;     ;; 	)
  ;;     (ros::sleep)
  ;;     )
  ;;   )
  )

(setq *people-pose* (instance Subscriber :init))
(send *people-pose* :ros-init)
