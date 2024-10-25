(load "package://msl_hand_controller/euslisp/jaxon_red-interface.l")
(load "package://auto_stabilizer/euslisp/auto-stabilizer-interface.l")

;;(unless (boundp '*jaxon_red*)
(jaxon_red-init)
(setq *robot* *jaxon_red*)
;;)
  
(objects (list *robot*))

;;(load "tutorial.l")
(load "package://control_tools/play-hrpsysfile.l")
(send *ri* :set-log-maxlength (* 500 120))
