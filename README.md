# Multi scale multi object template matching

This Python script demonstrates the use of OpenCV to find matching objects in an image using feature detection and matching techniques. The script supports two methods: Scale-Invariant Feature Transform (SIFT) and Oriented FAST and Rotated BRIEF (ORB).


## Features

- Detect and match objects in images using SIFT or ORB methods.
- Adjustable parameters for matching accuracy.
- Capability to handle multiple object matching in a single image.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Matplotlib (optional for visualization)

## Explanation

You can use SIFT or ORB to match templates with scale and rotation. After the first match, the matched area is filled with near neighbor pixels for the next match.


Image:  
![Alt text](i_remoter.png)  


Template:  
![Alt text](t_remoter.png)  

Matching:
![Alt text](results/iter1.png)

Remove matching area for next matching:  
![Alt text](results/iter1_.png)

Matching:
![Alt text](results/iter2.png)

and repeat matching...

Final result:  
![Alt text](results/output.png)  