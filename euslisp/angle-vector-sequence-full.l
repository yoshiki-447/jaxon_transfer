(defvar *wr-list* '())
(defun example ()
  (send *robot* :reset-pose)
  (send *ri* :angle-vector (send *robot* :angle-vector) 5000)
  (send *ri* :wait-interpolation)
  (setq av-list '())
  (setq tm-list '())
  (setq *wr-list* '())
  (dotimes (i 20)
    (setq diff (float-vector 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 i 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0))
    (push (v+ diff (send *robot* :angle-vector)) av-list)
    (push 10 tm-list)
    (push (apply #'float-vector (make-list 24 :initial-element 1.0)) *wr-list*)
    )
  (nreverse av-list)
  ;; (format t "av-list:~A~%" av-list)
  ;; (format t "tm-list:~A~%" tm-list)
  (send *ri* :angle-vector (send *robot* :angle-vector (car av-list)))
  (send *ri* :wait-interpolation)
  (send *ri* :angle-vector-sequence-full av-list tm-list :wrench *wr-list*)  ;;rarm-wrist-pをあげる
  (send *ri* :wait-interpolation)
  )