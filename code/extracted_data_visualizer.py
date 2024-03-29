import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

import matplotlib.animation as animation


CANDIDATE = int(input("candidate no: ") or '3')

df = pd.read_csv(rf'.\data\VID{CANDIDATE}_data.csv')
LIMIT = df.shape[0]
print(LIMIT)


#################
segments  = pd.read_csv(r"./train_videos/segments.csv", index_col="Train_no")
selection = segments.loc[CANDIDATE]

offsetF   = offsetS = 0
offset    = selection["offsetF"]
if offset < 0:
    offsetS = abs(offset)
else:
    offsetF = abs(offset)

camF = cv2.VideoCapture(fr".\train_videos\train{CANDIDATE}-f.mp4")
camS = cv2.VideoCapture(fr".\train_videos\train{CANDIDATE}-s.mp4")
    

#######################################

N = 0
prev_frame_no = 0

def update(frame):
    global COLS, N, prev_frame_no

    #################

    frame_no = df.iloc[N]['frame']
    if frame_no - prev_frame_no != 1:
        camF.set(cv2.CAP_PROP_POS_FRAMES, frame_no+offsetF-1)
        camS.set(cv2.CAP_PROP_POS_FRAMES, frame_no+offsetS-1)
    
    prev_frame_no = frame_no
    _, img_face    = camF.read()
    _, img_posture = camS.read()

    cv2.imshow('face', cv2.resize(img_face, (0,0), fx=0.3, fy=0.3))
    cv2.imshow('posture', cv2.resize(img_posture, (0,0), fx=0.3, fy=0.3))
    cv2.waitKey(1)
    #################


    if N == LIMIT: return
    
    br_x, br_y = [], []
    x,y = [], []
    ey_x, ey_y = [], []
    lp_x, lp_y = [], []

    posture_x, posture_y = [], []
    
    for col in ['brow_L', 'brow_R']:
        br_x.append( round(df.iloc[N][col + 'x'], 5) )
        br_y.append( 1 - round(df.iloc[N][col + 'y'], 5) )

    for col in ['face0', 'face1', 'face2', 'face3']:
        x.append( round(df.iloc[N][col + 'x'], 5) )
        y.append( 1 - round(df.iloc[N][col + 'y'], 5) )

    for eye in ['iris_L', 'iris_R']:
        ey_x.append( round(df.iloc[N][eye + 'x'], 5) )
        ey_y.append( 1 - round(df.iloc[N][eye + 'y'], 5) )

    for lp in ['lip_L',  'lip_R']:
        lp_x.append( round(df.iloc[N][lp + 'x'], 5) )
        lp_y.append( 1 - round(df.iloc[N][lp + 'y'], 5) )

    
    for col in ["shoulder", "knee"]:
        posture_x.append( round(df.iloc[N][col + 'x'], 5) )
        posture_y.append( 1 - round(df.iloc[N][col + 'y'], 5) )        


    # print(N)
    # update the scatter plot:
    data = np.stack([x, y]).T
    scat.set_offsets(data)

    data = np.stack([br_x, br_y]).T
    scat_brow.set_offsets(data)

    data = np.stack([ey_x, ey_y]).T
    scat_eye.set_offsets(data)

    data = np.stack([lp_x, lp_y]).T
    scat_lip.set_offsets(data)


    data = np.stack([posture_x, posture_y]).T
    posture.set_offsets(data)

    N += 1
    return (scat, scat_eye, scat_lip, scat_brow, posture)
    

#####################


fig, (ax_f, ax_s) = plt.subplots(1, 2)
fig.set_size_inches(12, 5)      

fig.suptitle("Extracted Features")
ax_f.title.set_text('Face')
ax_s.title.set_text('Posture')
ax_s.set_aspect(2)

center   = ax_f.scatter([0.5], [0.5], color='k', label="nose")
scat_brow= ax_f.scatter([0.4,0.5], [0.2,0.8], color='grey',   label='eyebrows')
scat_eye = ax_f.scatter([0.4,0.5], [0.2,0.8], color='brown',  label='iris')
scat_lip = ax_f.scatter([0.4,0.5], [0.2,0.8], color='pink',   label='lips')
scat     = ax_f.scatter([0.4,0.5], [0.2,0.8], color='orange', label='face')

center2  = ax_s.scatter([0.25], [0.45], color='k', label="hip")
posture  = ax_s.scatter([0.4,0.5], [0.2,0.8], color='orange', label='posture')


ax_f.set(xlim=[0, 1],
       ylim=[0, 1])
ax_f.set_xticks([0,0.5,1])
ax_f.set_yticks([0,0.5,1])
ax_f.legend()

ax_s.set(xlim=[0, 1])
ax_s.set_xticks([0,0.5,1])
ax_s.set_yticks([0,0.5,1])
ax_s.legend()


ani = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=2)

plt.show()


camF.release()
camS.release()
cv2.destroyAllWindows()
