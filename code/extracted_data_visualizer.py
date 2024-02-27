import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib.animation as animation


CANDIDATE = int(input("cnadidate no: ") or '1')

df = pd.read_csv(rf'.\data\VID{CANDIDATE}_data.csv')
LIMIT = df.shape[0]
print(LIMIT)


#################

N = 0

def update(frame):
    global COLS, N

    if N == LIMIT: return
    
    x,y = [], []
    ey_x, ey_y = [], []
    lp_x, lp_y = [], []
    
    for col in ['brow_L', 'brow_R', 'face0', 'face1', 'face2', 'face3']:
        x.append( round(df.iloc[N][col + 'x'], 5) )
        y.append( 1 - round(df.iloc[N][col + 'y'], 5) )

    for eye in ['iris_L', 'iris_R']:
        ey_x.append( round(df.iloc[N][eye + 'x'], 5) )
        ey_y.append( 1 - round(df.iloc[N][eye + 'y'], 5) )

    for lp in ['lip_L',  'lip_R']:
        lp_x.append( round(df.iloc[N][lp + 'x'], 5) )
        lp_y.append( 1 - round(df.iloc[N][lp + 'y'], 5) )
        


    print(N)
    # update the scatter plot:
    data = np.stack([x, y]).T
    scat.set_offsets(data)

    data = np.stack([ey_x, ey_y]).T
    scat_eye.set_offsets(data)

    data = np.stack([lp_x, lp_y]).T
    scat_lip.set_offsets(data)
    N += 1
    return (scat, scat_eye, scat_lip)

#####################


fig, ax = plt.subplots()


center   = ax.scatter([0.5], [0.5], color='k')
scat     = ax.scatter([0.4,0.5], [0.2,0.8], color='orange', label='face')
scat_eye = ax.scatter([0.4,0.5], [0.2,0.8], color='brown',  label='iris')
scat_lip = ax.scatter([0.4,0.5], [0.2,0.8], color='pink',   label='lips')


ax.set(xlim=[0, 1],
       ylim=[0, 1])
ax.legend()


ani = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=2)

plt.show()
