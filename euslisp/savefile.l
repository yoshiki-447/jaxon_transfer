(defun save-file (filename data)
  (let ((file (open filename :direction :output)))
    (when file
      (format file "~A~%" data)
      (close file)
      )
    )
  )
  
(defun read-file (filename)
  (let ((file (open filename :direction :input))
	(data '()))
    (when file
      (do ((item (read file nil) (read file nil)))
	  ((null item))
	  (setq data item)
	  ;;(setq data (append data (list item)))
	  )
      ;; (loop for item = (read file nil)
      ;; 	    while item do
	    ;; (cond ((> (length data) 0)
	    ;; 	   (nconc data (list item))
	    ;; 	   )
	    ;; 	  (t
	    ;; 	   (push item data)
	    ;; 	   )
	    ;; 	  )
	   ;; (setq data (append data (list item)))
;;	    )
      (close file)
      )
    data
    )
  )

;; save-file "../data/output.txt" b
;;setq data (read-file "../data/output.txt")
