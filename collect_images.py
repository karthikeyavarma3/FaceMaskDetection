import numpy as np
import os
import cv2
import time



def collect_images(dir , categories , no_of_images):

    cap = cv2.VideoCapture(0)

    

    for category in categories:

        

        

        img_count = 0

        print(f'Collecting dataset of categ : {category}')

        while True:

            ret , frame = cap.read()

            cv2.imshow("webcam" , frame)
            
            key = cv2.waitKey(1)

            if key == ord('q'):
                break
            elif key == ord('s'):
                path = os.path.join(dir , category , f'img_{img_count}.png')
                save_img = cv2.imwrite(path , frame)
                print(f'saved img_{img_count}')
                img_count +=1

                time.sleep(1)

                if img_count == no_of_images:
                    break

    
    cap.release()
    cv2.destroyAllWindows()


categories = ['Facemask' , 'Incorrect' , 'No Face-mask']

parent_dir = r'C:\Projects\Dataset'

images_count = 15

collect_images(parent_dir , categories , images_count)