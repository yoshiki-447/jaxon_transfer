(load "package://eus_prioritized_inverse_kinematics/euslisp/prioritized-inverse-kinematics-compiled.l")

(defun example (&key(larm-trans-pos #f(0 0 0)) (larm-trans-rpy #f(0 0 0)) (elbow-trans-pos #f(0 0 0)) (elbow-trans-rpy #f(0 0 0)) &rest args)
  ;; (let* ((larm-end-coords (send *robot* :larm :end-coords :copy-worldcoords))
  ;; 	 (larm-elbow-coords (send *robot* :larm_link4_lk :copy-worldcoords))
  ;; 	 (larm-rpy (apply #'float-vector (car (send (send *robot* :larm :end-coords :copy-worldcoords) :rpy-angle))))
  ;; 	 (elbow-rpy (apply #'float-vector (car (send (send *robot* :larm_link4_lk :copy-worldcoords) :rpy-angle))))
  ;; 	 (larm-target (send larm-end-coords :translate larm-trans-pos :world) :copy-worldcoords)
  ;; 	 (elbow-target (send larm-elbow-coords :translate elbow-trans-pos :world) :copy-worldcoords))
  (setq larm-target (make-cascoords :coords (send *robot* :larm :end-coords :copy-worldcoords)))
  (send larm-target :translate larm-trans-pos :world)
  (send larm-target :transform (make-coords :pos #f(0 0 0) :rpy larm-trans-rpy) :local)
  (setq elbow-target (make-cascoords :coords (send *robot* :larm_link4_lk :copy-worldcoords)))
  (send elbow-target :translate elbow-trans-pos :world)
  (send elbow-target :transform (make-coords :pos #f(0 0 0) :rpy elbow-trans-rpy) :local)
  (apply #'prioritized-inverse-kinematics
	   (list 
	    (list (instance move-target-task :init
			    (send *robot* :larm :end-coords)
			    larm-target
			    :thre 2.0
			    :rthre 0.1
			    :check nil))
	    (list (instance move-target-task :init
			    (send *robot* :larm_link4_lk)
			    ;; lelbow-target
			    elbow-target
			    :translation-axis t
			    ;; :rotation-axis nil
			    :check nil)))
	   (append
	    ;; (list (instance virtual-joint-variable :init *robot*))
	    (mapcar #'(lambda (j) (instance joint-variable :init j))
		    (send *robot* :larm :joint-list)))
	   :task0 (append
		   ;; (list (instance minmax-angle-task :init *robot*
		   ;; :min-angle (float-vector -200 -200 -200 -20 -20 -10)
		   ;; :max-angle #F(200 200 200 20 20 10)))
		   (mapcar #'(lambda (j) (instance minmax-angle-task :init j))
			   (send *robot* :larm :joint-list)))
	   :min-loop 50
	   :debug-view nil
	   args)
  ;;)
  )

(defun arm-solve-ik
    (&key(rarm-trans-pos #f(0 0 0))
	 (larm-trans-pos #f(0 0 0)))
	 ;; (rarm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad 0)))
	 ;; (larm-rpy (float-vector (deg2rad 0) (deg2rad 0) (deg2rad 0))))
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
	  ;; :null-space #f(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
	  ;; :thre (list 2.0 2.0)
	  ;; :rthre (list 0.1 0.1)
	  ;; :target-centroid-pos (midpoint 0.5
	  ;; 				 (send *robot* :rleg :end-coords :worldpos)
	  ;; 				 (send *robot* :rleg :end-coords :worldpos))
	  ;; :cog-translation-axis :z
	  )
    )
  )
