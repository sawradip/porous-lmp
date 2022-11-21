import numpy as np
import  os
import cv2
  
def isOverlapping(xy, prev_positions_list, radius):
    for pos in prev_positions_list:
        dist = np.linalg.norm(xy - pos)
        if dist < radius:
            return True
    return False

def maskgen(PORE_PERCENTAGE = None,
            PORE_NUMBER = None,
            THRESHOLD = 20,
            AVG_PORE_DIA = None,  
            MODE = 'ELLIPSE',  
            ELLIPSE_RANGE_PERCENTAGE = None,  
            OUT_IMSHAPE = (1000, 1000),
            OUT_FOLDER = "porous",
            file_name = 'pore_10pr_5'
            ):
    # path 
    if not os.path.exists(OUT_FOLDER):
        os.mkdir(OUT_FOLDER)
    path = r'black_circle.png'
  
    image = cv2.imread(path)
    circle = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    circle[circle <= THRESHOLD] = 0
    circle[circle > THRESHOLD] = 1

    if not PORE_NUMBER:
        if not PORE_PERCENTAGE:
            raise ValueError('PORE_PERCENTAGE must be defined if no PORE_NUMBER given')
        else:
            TOTAL_AREA = OUT_IMSHAPE[0] * OUT_IMSHAPE[1]
            TOTAL_PORE_AREA = PORE_PERCENTAGE * TOTAL_AREA
            SINGLE_PORE_AREA = (np.pi / 4)* AVG_PORE_DIA**2
            PORE_NUMBER = int(TOTAL_PORE_AREA // SINGLE_PORE_AREA )
      


    if MODE == 'ELLIPSE' and ELLIPSE_RANGE_PERCENTAGE:
        ELLIPSE_RANGE = int(AVG_PORE_DIA * ELLIPSE_RANGE_PERCENTAGE)
        MAX_OFFSET = AVG_PORE_DIA + ELLIPSE_RANGE
        OVERLAP_THRESHOLD = AVG_PORE_DIA + ELLIPSE_RANGE
    elif MODE == 'CIRCLE':
        MAX_OFFSET = AVG_PORE_DIA
        OVERLAP_THRESHOLD = AVG_PORE_DIA
    else:
        raise ValueError('MODE must be one of "ELLIPSE" or "CIRCLE"')



# MODE = 'CIRCLE'  





# PORE_NUMBER = 20


# Random Coord Generation
# if MODE == 'CIRCLE':
# elif MODE == 'ELLIPSE':
    


# Overlapping Threshold
# if MODE == 'CIRCLE':
# elif MODE == 'ELLIPSE':

# rand_x = np.random.randint(low = MAX_OFFSET, high = (OUT_IMSHAPE[0] - MAX_OFFSET), size = PORE_NUMBER)
# rand_y = np.random.randint(low = MAX_OFFSET, high = (OUT_IMSHAPE[1] - MAX_OFFSET), size = PORE_NUMBER)

    bg_image = np.ones(OUT_IMSHAPE) * 255

    pore_counter = 0
    prev_position_list = []
    
    rand_x = np.zeros(PORE_NUMBER, dtype=int)
    rand_y = np.zeros(PORE_NUMBER, dtype=int)

    for i in range(PORE_NUMBER):
        rand_xy = np.random.randint(low = MAX_OFFSET, high = (OUT_IMSHAPE[0] - MAX_OFFSET), size = 2)
        
        MAX_LOOP = 500
        loop_counter = 0
        while isOverlapping(rand_xy, prev_position_list, OVERLAP_THRESHOLD) and loop_counter < MAX_LOOP:
            rand_xy = np.random.randint(low = MAX_OFFSET, high = (OUT_IMSHAPE[0] - MAX_OFFSET), size = 2)   
            loop_counter = loop_counter + 1

        prev_position_list.append(rand_xy)
        rand_x[i] = rand_xy[0]
        rand_y[i] = rand_xy[1]




    for n in range(PORE_NUMBER):

        if MODE == 'CIRCLE':
            circle_w = AVG_PORE_DIA
            circle_h = AVG_PORE_DIA
        elif MODE == 'ELLIPSE':
            circle_w = AVG_PORE_DIA + np.random.randint(ELLIPSE_RANGE)
            circle_h = AVG_PORE_DIA - np.random.randint(ELLIPSE_RANGE)

        s_circle = cv2.resize(circle , (circle_w , circle_h), interpolation = cv2.INTER_AREA)

        y1, y2 = rand_y[n] - circle_h//2, rand_y[n] - circle_h//2 + s_circle.shape[0]
        x1, x2 = rand_x[n] - circle_w//2, rand_x[n] - circle_w//2 + s_circle.shape[1]

        bg_image[y1:y2, x1:x2] = bg_image[y1:y2, x1:x2] * s_circle

    
    binArrayFunc = lambda x:  1 if x < THRESHOLD else 0
    npBinArrayFunc = np.vectorize(binArrayFunc)

    bin_image = npBinArrayFunc(bg_image)

    out_txt_path = os.path.join(OUT_FOLDER, f'{file_name}.txt')
    out_img_path = os.path.join(OUT_FOLDER, f'{file_name}.jpg')

    with open( out_txt_path, 'w') as f:
        for line in bin_image.astype(str):
            line = ('').join(list(line))
            f.write(line)
            f.write('\n')
    cv2.imwrite( out_img_path, bg_image)

    return bin_image



maskgen(PORE_PERCENTAGE = 0.20,
        MODE='ELLIPSE',
        AVG_PORE_DIA= 20,
        ELLIPSE_RANGE_PERCENTAGE=0.25,
        OUT_IMSHAPE = (300, 300),
        OUT_FOLDER = "porous_20pr",
        file_name = 'pore_20pr_5'
        )
