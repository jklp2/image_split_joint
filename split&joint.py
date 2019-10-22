import numpy as np
import cv2

def split(img,nh,nw,overlap=20):  #img(1*3*h*w) nh parts for heights, nw parts for width
    _,_,h,w = img.shape
    ph = [0]
    pw = [0]
    for i in range(nh - 1):
        ph.append(ph[-1] + round(h / nh))
    ph.append(h - 1)
    for i in range(nw - 1):
        pw.append(pw[-1] + round(w / nw))
    pw.append(w - 1)
    sw = []
    sh = []
    for i in range(nh):
        sh.append([max(0, ph[i] - overlap), min(h, ph[i + 1] + overlap)])
    for i in range(nw):
        sw.append([max(0, pw[i] - overlap), min(w, pw[i + 1] + overlap)])
    img_slice = []
    for i in range(nh):
        for j in range(nw):
            img_slice.append(img[:, :, sh[i][0]:sh[i][1], sw[j][0]:sw[j][1]])
    return img_slice,sh,sw

def joint(img_slice,sh,sw):
    nw=len(sw)
    nh=len(sh)
    w=sw[-1][1]
    h=sh[-1][1]
    rec = np.zeros((1, 3, h, w))
    cont = np.zeros((h, w))
    for i in range(nh):
        for j in range(nw):
            cur = img_slice[i * nw + j]
            for ii in range(sh[i][0], sh[i][1]):
                for jj in range(sw[j][0], sw[j][1]):
                    cont[ii, jj] += 1
                    rec[0, :, ii, jj] += cur[0, :, ii - sh[i][0], jj - sw[j][0]]
    for i in range(h):
        for j in range(w):
            if cont[i][j]!=1:
                rec[0, :, i, j]/=cont[i][j]
    return rec

if __name__=='__main__':
    img_path = "./1.png"
    img = cv2.imread(img_path)/255
    img = np.array([np.transpose(img,[2, 0, 1])])   #1*3*h*w

    img_slice, sh, sw = split(img,1,1)
    rec = joint(img_slice, sh, sw)

    rec=np.transpose(rec[0], (1, 2, 0))
    cv2.imshow("0", rec)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.destroyWindow("0")
