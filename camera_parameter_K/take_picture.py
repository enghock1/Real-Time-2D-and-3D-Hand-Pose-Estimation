import cv2
import matplotlib.pyplot as plt
import time

cap = cv2.VideoCapture(0)

num_image = 30

for i in range(num_image):

    ret, frame = cap.read()

    plt.imshow(frame)
    plt.axis('off')
    
    img_name = 'img'+str(i)
    plt.savefig('calibration_images/'+img_name+'.png', bbox_inches='tight')
    plt.show()
    
    start = time.time()
    
    while time.time() - start < 3:
        pass
    

        
cap.release()
cv2.destroyAllWindows()



























