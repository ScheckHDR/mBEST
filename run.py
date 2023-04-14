import sys
from mBEST import mBEST
from mBEST.color_masks import *

def crop_rope(img,extra:int=10,return_roi:bool=False):
    coords = np.argwhere(img[:,:,2] < 50)
    mins = np.min(coords,axis=0) - extra
    maxs = np.max(coords,axis=0) + extra

    ymin,xmin = np.clip(mins,0,img.shape[:2])
    ymax,xmax = np.clip(maxs,0,img.shape[:2])

    if return_roi:
        return img[ymin:ymax+1,xmin:xmax+1,:],((ymin,xmin),(ymax,xmax))
    else:
        return img[ymin:ymax+1,xmin:xmax+1,:]

def get_mask(img):
    mask = ((img[:,:,2] < 50) * 255).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

dlo_seg = mBEST(get_mask, epsilon=40, delta=25)

img = cv2.imread("/home/jeffrey/Pictures/test0.png")
img = crop_rope(img)

# paths = dlo_seg.run(mask)
# dlo_seg.run(img, plot=True)
dlo_seg.run(img, intersection_color=[255, 0, 0], plot=True)
