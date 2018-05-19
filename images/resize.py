import glob
import cv2

for fname in glob.glob('bad_convergence.png'):
    img = cv2.imread(fname)
    dst = cv2.resize(img, (0,0), fx=0.3, fy=0.3)
    cv2.imwrite('r_' + fname, dst)

print('done')
    
