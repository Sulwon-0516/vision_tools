'''
Utility functions for EDA.


'''

import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from math import floor


my_dpi = 100    # matplotlib imsave option

def cv2plt(img):
    '''
    convert cv2 img into plt
    '''
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mat_img = img / 255.0

    return mat_img


def plt2cv(img):
    '''
    convert plt into cv2
    '''
    img = int(img*255)

    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def PIL2cv(img):
    '''
    convert Pillow img into cv2
    '''
    opencv_image=np.array(img)

    if len(img.shape) == 3 and img.shape[2] == 3:
        opencv_image=cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)

    return opencv_image

def cv2PIL(img):
    '''
    convert cv2 img into Pillow
    '''
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_image=Image.fromarray(img)

    return pil_image



def plot_bbox(inp, xmin, xmax, ymin, ymax):
    '''
    plot bbox.
    It only gets "BGR cv" image input
    '''
    pts1 = (xmin, ymin)
    pts2 = (xmax, ymax)

    cv2.rectangle(inp, pts1, pts2, color=(0, 255, 0), thickness=3)

    return inp


def my_subplots(imgs, n_cols = 3, n_rows = 3, suptitle="", titles = None):
    '''
    Auto subplot
    N_x : # of images on x-direction
    N_y : # of images on y-direction
    '''
    n_imgs = len(imgs)
    if n_cols * n_rows < n_imgs:
        print("# of images > # of slots to plot, imgs: %d, slots: %d"%(n_imgs, n_cols*n_rows))

    # process titles
    if isinstance(titles, type(None)):
        titles = ["" for i in range(n_imgs)]
    elif len(titles) < n_imgs:
        n_wo_title = n_imgs - len(titles)
        for i in range(n_wo_title):
            titles.append("")

    fig, axs = plt.subplots(n_cols, n_rows, figsize = (n_rows*250/my_dpi, (n_cols+1)*250/my_dpi), dpi = my_dpi)
    fig.suptitle(suptitle, fontsize = 40)

    ind = 0
    for img in imgs:
        col = ind%n_rows
        row = ind//n_rows

        if ind == n_cols * n_rows:
            break

        ax = axs[row, col]
        ax.axis('off')
        ax.imshow(img)
        ax.set_title(titles[ind], fontsize = 15)
        ind+=1
    
    return fig





def rand_color_gen():
    # make color in "0~256)
    r = np.random.uniform(0,1)
    g = np.random.uniform(0,1)
    b = np.random.uniform(0,1)

    inten = np.random.uniform(0.2,0.8)
    ratio = 255 * inten

    color = (int(floor(r*ratio)), int(floor(g*ratio)), int(floor(b*ratio)))

    return color
    

def rand_N_color_gen(N_color):
    colors = []

    for i in range(N_color):
        c = rand_color_gen()

        if c in colors:
            # when it's duplicated
            while(c in colors):
                c = rand_color_gen()
        
        colors.append(c)
    
    return colors



def freq_pixel_clustering(img, N_cluster = 4):
    colors = rand_N_color_gen(N_cluster)

    if len(img.shape) == 2:
        img = np.expand_dims(img, axis = 2)

    if len(img.shape) != 3:
        print("img should be 3-D")
        assert()
    
    H, W, _ = img.shape

    pixel_list = []
    pixel_cnt = []

    # count # of cnts
    for x in range(H):
        for y in range(W):
            pixel = img[x,y,:]

            if pixel in pixel_list:
                pixel_cnt[pixel_list.index(pixel)] += 1
            else:
                pixel_list.append(pixel)
                pixel_cnt.append(1)
    
    # sort pixels
    sort_ind = sorted(range(len(pixel_cnt)), key = lambda k: pixel_cnt[k])
    N_cluster = len(sort_ind) if N_cluster > len(sort_ind) else N_cluster

    stnd_ind = sort_ind[0:N_cluster]
    stnd_pixel = [pixel_list[ind] for ind in stnd_ind]

    # change
    clustered = np.zeros((H,W,3), dtype = np.int64)
    for x in range(H):
        for y in range(W):
            pixel = img[x,y,:]

            if pixel in stnd_pixel:
                ind = stnd_pixel.index(pixel)
                c = colors[ind]
                clustered[x,y,:] = np.array(list(c))
            
            else:
                # find closest pixel
                min_dist = 10000
                min_ind = 0
                i = 0
                for p in stnd_pixel:
                    d = ((pixel - p)**2).sum()
                    if d < min_dist:
                        min_dist = d
                        min_ind = i
                    i+=1
                
                c = colors[min_ind]
                clustered[x,y,:] = np.array(list(c))
    
    return clustered
                    



    
            


    



