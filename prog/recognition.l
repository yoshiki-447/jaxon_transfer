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

(defun recognition (msg string1 string2)
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
	  (format t "~A index: ~A~%" string2 shoulder-index)
	  (let* ((coords1 (make-coords :pos (ros::tf-point->pos (send (elt (send pose-msg :bones) shoulder-index) :end_point))
				       :rpy (float-vector 0 0 0)))
		 (coords2 (world-to-target coords1)))
	    (setq coords coords2)
	    ;;(print coords)
	    (setq pos (send coords :pos))
	    (print (send coords1 :pos))
	    (print pos)
	    pos
	    )
	  )
	)
      )
   )

(setq *r-shoulder-list* '())
(setq *l-elbow-list* '())

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
    (setq r-shoulder-pos nil)
    (setq l-elbow-pos nil)
    (setq r-shoulder-pos (recognition msg "left shoulder->right shoulder" "right-shoulder"))
    (setq l-elbow-pos (recognition msg "right shoulder->left elbow" "left-elbow"))		
		
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
      (when (and r-shoulder-pos l-elbow-pos)
	(format t "norm: ~A~%" (norm r-shoulder-pos))
	(format t "norm: ~A~%" (norm l-elbow-pos))
	(let* ((limit 600)
	       (dist-check-r-shoulder (< (norm r-shoulder-pos) limit))
	       (dist-check-l-elbow (< (norm l-elbow-pos) limit)))
	  (cond ((and dist-check-r-shoulder dist-check-l-elbow)
		 (arm-solve-ik-locate :larm-trans-pos r-shoulder-pos)
		 (arm-solve-ik-locate :rarm-trans-pos l-elbow-pos)
		 (print "success")
		 )
		(t
		 (format t "ik failed~%")
		 )
		)
	  )
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
