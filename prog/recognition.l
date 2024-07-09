(ros::roseus "recognition")
(ros::roseus-add-msgs "jsk_recognition_msgs")

;; (ros::rate 20)
;; (:callback (msg)
;; 	   (setq 
;; (ros::subscribe "/edgetpu_human_pose_estimator/output/poses" jsk_recognition_msgs::PeoplePoseArray #'send self:callback 1)

(defun world-to-target (target)
  (let target-coords
    ;; ikを解く前にrobotに実機の姿勢を反映
    (send *robot* :angle-vector (send *ri* :state :potentio-vector))
    ;; (send robot :fix-leg-to-coords (make-coords))
    (setq target-coords
      (send (send *robot* :rs_l515_depth_optical_frame :copy-worldcoords) :transform target))
    target-coords)
  )

(defclass Subscriber
  :super propertied-object
  :slots (limb-msg pose-msg shoulder-index r-shoulder-coords)
  )

(defmethod Subscriber
  (:init () (setq limb-msg nil
		  pose-msg nil
		  shoulder-index nil
		  r-shoulder-coords nil)
	 self)
  
  (:ros-init ()
    (ros::roseus "sub")
    (ros::subscribe "/edgetpu_human_pose_estimator/output/poses" ;; subscribeするtopicの名前を指定
            jsk_recognition_msgs::PeoplePoseArray ;; msgの型
            #'send self :cb-function ;; call-backメソッドを指定
            )
   )
  
  (:cb-function (msg)
    (when (> (length (send msg :poses)) 0) ;; 一人以上認識したら処理をする
      (setq pose-msg (car (send msg :poses)))
      (setq limb-msg (send pose-msg :limb_names))
      (let ((count 0) (success nil))
        (dolist (limb-name limb-msg)
	  (when (string-equal limb-name "right shoulder")
	    (setq shoulder-index count)
	    (setq success t)
	    )
	  (incf count)
	  )
	(when success
	  (format t "right-shoulder index: ~A~%" shoulder-index)
	  (setq r-shoulder-coords (ros::tf-pose->coords  (elt (send pose-msg :poses) shoulder-index)))
	  (print r-shoulder-coords)
	  )
	
      ;; (print (car (send pose-msg :poses)))
	)
      )
    )

  (:ros-motion ()
    (while (ros::ok)
      (ros::spin-once)
      ;; ここにsubscribeした結果を元にした動作メソッドとかを入れたりする
      (ros::sleep)
      )
    )
  )

(setq *people-pose* (instance Subscriber :init))

