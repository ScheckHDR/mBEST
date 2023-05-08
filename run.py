import sys
from mBEST import mBEST
from mBEST.color_masks import *
import topology

def crop_rope(img,extra:int=10,return_roi:bool=False):
    kernel = np.ones((13,13), np.uint8)
    mask = get_mask(img)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    coords = np.argwhere(mask)

    mins = np.min(coords,axis=0) - extra
    maxs = np.max(coords,axis=0) + extra

    ymin,xmin = np.clip(mins,0,img.shape[:2])
    ymax,xmax = np.clip(maxs,0,img.shape[:2])

    if return_roi:
        return img[ymin:ymax+1,xmin:xmax+1,:],((ymin,xmin),(ymax,xmax))
    else:
        return img[ymin:ymax+1,xmin:xmax+1,:]

def get_mask(img):
    mask = ((np.all(img > 200,axis=2)) * 255).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

dlo_seg = mBEST(get_mask, epsilon=40, delta=25)

img = cv2.imread("/home/jeffrey/Downloads/rope_imgs/3.jpg")[500:-500,:,:]
img = cv2.resize(img,(720,720))
img = crop_rope(img)

# paths = dlo_seg.run(mask)
# dlo_seg.run(img, plot=True)
paths, overs = dlo_seg.run(img, intersection_color=[255, 0, 0], plot=0,overlay=False)

# for i in range(len(paths)):
#     # remove duplicates
#     _,unique_indices = np.unique(paths[i],return_index=True,axis=0)
#     unique_indices = np.sort(unique_indices)
#     p = paths[i][unique_indices,:]
#     o = overs[i][unique_indices]
#     # p = paths[i][:,:]
#     # o = overs[i][:]
#     p = np.insert(p,1,o,axis=1)
#     a = topology.RopeTopology.from_geometry(p[::1,::-1])
#     print(a.rep)