(load "models/room602-wheelchair-object.l")
(load "models/room602-bed-object.l")
(load "irteus/demo/sample-robot-model.l")

(setq *chair* (room602-wheelchair))
(setq *bed* (room602-bed))
(setq *human* (instance sample-robot :init))

(send *chair* :move-to (make-coords :pos :rpy))
