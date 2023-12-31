import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
from numba import njit

dtheta = 10
num_theta = 100000
fig, ax = plt.subplots(figsize=(9,8))
plt.subplots_adjust(left=0.5)

@njit(cache=True)
def ellipse(theta, a=1., b=1., rot=0., x0=0., y0=0.):
    x = x0 + a * np.cos(theta) * np.cos(rot) - b * np.sin(theta) * np.sin(rot)
    y = y0 + a * np.cos(theta) * np.sin(rot) + b * np.sin(theta) * np.cos(rot)
    return x, y

@njit(cache=True)
def run_draw(P,T,dtheta,num_theta):
    xs,ys = [],[]
    theta_max = num_theta*np.pi
    thetas = np.arange(0,theta_max,dtheta)
    for theta in thetas:
        scale = theta/theta_max
        x0_0,y0_0 = ellipse(scale/(P[6]+0.01)/100,P[0]*scale,P[1],P[2])
        x0_1,y0_1 = ellipse(scale*P[7]/10,P[3]*scale,P[4]*scale,P[5]*scale)
        x0,y0 = x0_0+x0_1,y0_0+y0_1
        x1,y1 = ellipse(theta/T[0]+T[0],P[0]*scale,P[1],P[2],x0,y0)
        x2,y2 = ellipse(theta/T[1]+T[1],P[3]*scale,P[4],P[5],P[6]*x1*scale,P[7]*y1*scale)
        x,y = x1+x2,y1+y2
        xs.append(x)
        ys.append(y)
    return xs,ys

@njit(cache=True)
def make_params():
    P = np.random.uniform(0,10,9)
    T = np.random.randint(1, 1000, 2)
    return P,T

def animate(event):
    P = np.array([s.val for s in ss[:8]])
    T = np.array([s.val for s in ss[8:10]])
    num_theta = ss[10].val
    fig, _ = plt.subplots(figsize=(9,8))
    xs,ys = run_draw(P,T,dtheta,num_theta)
    l, = plt.plot(xs, ys, lw=0.5)
    plt.axis('off')
    def update(i):
        l.set_data(xs[:i],ys[:i])
        return l,

    def frames():
        ind=0
        speed = 50
        # Scale by constants related to sin(x^3)
        x_denom = len(xs)/1.4646
        rescale = 0.4034988/speed
        for i in range(len(xs)//speed):
            print(f'{round(100*speed*i/len(xs)):3}%',end='\r')
            ind += np.sin(speed**3*i**3/x_denom**3)/rescale
            yield int(ind)

    l.set_data(xs,ys)
    sc = len(list(frames()))
    anim = FuncAnimation(fig, update, frames=frames, save_count=sc, interval=0,
                         blit=True, repeat=False)
    return anim

def animshow(event):
    anim = animate(event)
    plt.show()

def animsave(event):
    anim = animate(event)
    anim.save('output.mp4', writer='ffmpeg', fps=30, codec='h264')
    print('Done')

def reset(event):
    [s.reset() for s in ss]

def update(val):
    P = np.array([s.val for s in ss[:8]])
    T = np.array([s.val for s in ss[8:10]])
    num_theta = ss[10].val
    xs,ys = run_draw(P,T,dtheta,num_theta)
    l.set_data(xs,ys)
    fig.canvas.draw_idle()
    ax.relim()
    ax.autoscale_view()

P = np.array([2.28733802, 5.46057411, 1.51468886, 2.48589304, 7.68418836, 1.73164839,
    1.37929407, 6.90116276, 4.30064857])
T = np.array([326., 325.])
xs,ys = run_draw(P,T,dtheta,num_theta)
l, = plt.plot(xs, ys, lw=0.5)
ax.set_aspect('equal')
ax.axis('off')
axcolor = 'gainsboro'
axs = [plt.axes([0.05, 8*i/100+0.07, 0.3, 0.03], facecolor=axcolor) for i in range(11)]
ss = [Slider(ax, 'P'+str(i), 0, 10, valinit=P[i]) for i,ax in enumerate(axs[:8])]
ss.append(Slider(axs[8], 'T0', T[0]-5, T[0]+5, valinit=T[0]))
ss.append(Slider(axs[9], 'T1', T[1]-5, T[1]+5, valinit=T[1]))
ss.append(Slider(axs[10], 'N', 1, 1000000, valinit=num_theta))

[s.on_changed(update) for s in ss]

res_but= Button(plt.axes([0.85, 0.9, 0.1, 0.04]), 'Reset', color=axcolor, hovercolor='0.975')
anishow_but = Button(plt.axes([0.67, 0.9, 0.15, 0.04]), 'Show Animate', color=axcolor, hovercolor='0.975')
anisave_but = Button(plt.axes([0.50, 0.9, 0.15, 0.04]), 'Save Animate', color=axcolor, hovercolor='0.975')

res_but.on_clicked(reset)
anishow_but.on_clicked(animshow)
anisave_but.on_clicked(animsave)

plt.show()


