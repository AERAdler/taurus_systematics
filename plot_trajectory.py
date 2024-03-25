import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from matplotlib import colormaps
import healpy as hp
import cartopy.crs as ccrs
import shapely.geometry as sgeom
import cartopy.io.shapereader as shpreader

plt.rc("xtick", labelsize=9)
plt.rc("ytick", labelsize=9)

fig, ax = plt.subplots(2, figsize=(10/3, 3), sharex=True)

track1 = np.loadtxt("balloontrack1.txt", unpack=True)
track2 = np.loadtxt("balloontrack2.txt", unpack=True)
track3 = np.loadtxt("balloontrack3.txt", unpack=True)

t1 = track1[0,0]
time1 = (track1[0]-t1)/(3600*24)

bpts = np.array(np.nonzero(np.abs(track1[2,1:]-track1[2,:-1])>180))+1
bpts = np.insert(bpts, 0, [0])
bpts = np.append(bpts, [track1.shape[1]])

for i in range(bpts.size-1):
    ax[0].plot(time1[bpts[i]:bpts[i+1]], track1[2,bpts[i]:bpts[i+1]], c="tab:blue")
ax[1].plot(time1, track1[1], label="Qualification")

t2 = track2[0,0]
time2 = (track2[0]-t2)/(3600*24)
bpts = np.array(np.nonzero(np.abs(track2[2,1:]-track2[2,:-1])>180))+1
bpts = np.insert(bpts, 0, [0])
bpts = np.append(bpts, [track2.shape[1]])

for i in range(bpts.size-1):
    ax[0].plot(time2[bpts[i]:bpts[i+1]], track2[2,bpts[i]:bpts[i+1]], c="tab:orange")
ax[1].plot(time2, track2[1], label="COSI")

t3 = track3[0,0]
time3 = (track3[0]-t3)/(3600*24)
bpts = np.array(np.nonzero(np.abs(track3[3,1:]-track3[3,:-1])>180))+1
bpts = np.insert(bpts, 0, [0])
bpts = np.append(bpts, [track3.shape[1]])

for i in range(bpts.size-1):
    ax[0].plot(time3[bpts[i]:bpts[i+1]], -track3[3,bpts[i]:bpts[i+1]], c="tab:green")
ax[1].plot(time3, track3[2], label="SuperBIT")

ax[0].set_ylabel("Longitude (°)", fontsize=9)
ax[1].set_ylabel("Latitude (°)", fontsize=9)
ax[1].set_xlabel("Time from launch (days)", fontsize=9)
ax[0].tick_params(axis="x", direction="in")
ax[0].set_ylim(-180, 180)
ax[0].set_xlim(0, 51)
fig.legend(frameon=False, fontsize=9, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.915))
plt.tight_layout(h_pad=0.2)
plt.savefig("trajectory.pdf", bbox_inches="tight", dpi=180)

### Trajectory on map
fig2, ax2 = plt.subplots(figsize=(10/3, 10/3), 
    subplot_kw={"projection": ccrs.SouthPolarStereo()})
ax2.set_extent([-180, 180, -90, -10], crs=ccrs.PlateCarree())
# Add map features (optional)
ax2.coastlines()
ax2.gridlines()
cmap_tracks = colormaps["tab20"]
cflip = 0
colors = [cmap_tracks(0), cmap_tracks(0.05)]
for d in range(int(time1[-1])+1):

    daystart = np.argmin(np.abs(time1-d))
    dayend = np.argmin(np.abs(time1-d-1))
    daytrack = track1[:, daystart:dayend+1]
    bpts = np.array(np.nonzero(np.abs(daytrack[2,1:]-daytrack[2,:-1])>180))+1
    bpts = np.insert(bpts, 0, [0])
    bpts = np.append(bpts, [daytrack.shape[1]])

    for i in range(bpts.size-1):
        ax2.plot(-daytrack[2, bpts[i]:bpts[i+1]], daytrack[1, bpts[i]:bpts[i+1]],
            c=colors[cflip], transform=ccrs.PlateCarree())

    cflip+=1
    cflip = cflip%2
ax2.plot(-track1[2, dayend:], track1[1, dayend:],
            c=colors[cflip], transform=ccrs.PlateCarree())

cflip = 0
colors = [cmap_tracks(0.1), cmap_tracks(0.15)]
for d in range(int(time2[-1])+1):

    daystart = np.argmin(np.abs(time2-d))
    dayend = np.argmin(np.abs(time2-d-1))
    daytrack = track2[:, daystart:dayend+1]
    bpts = np.array(np.nonzero(np.abs(daytrack[2,1:]-daytrack[2,:-1])>180))+1
    bpts = np.insert(bpts, 0, [0])
    bpts = np.append(bpts, [daytrack.shape[1]])

    for i in range(bpts.size-1):
        ax2.plot(-daytrack[2, bpts[i]:bpts[i+1]], daytrack[1, bpts[i]:bpts[i+1]],
            c=colors[cflip], transform=ccrs.PlateCarree())

    cflip+=1
    cflip = cflip%2
ax2.plot(-track2[2, dayend:], track2[1, dayend:],
            c=colors[cflip], transform=ccrs.PlateCarree())


cflip = 0
colors = [cmap_tracks(0.2), cmap_tracks(0.25)]
for d in range(int(time3[-1])+1):
    daystart = np.argmin(np.abs(time3-d))
    dayend = np.argmin(np.abs(time3-d-1))
    daytrack = track3[:, daystart:dayend+1]
    bpts = np.array(np.nonzero(np.abs(daytrack[3,1:]-daytrack[3,:-1])>180))+1
    bpts = np.insert(bpts, 0, [0])
    bpts = np.append(bpts, [daytrack.shape[1]])
    for i in range(bpts.size-1):
        ax2.plot(daytrack[3, bpts[i]:bpts[i+1]], daytrack[2, bpts[i]:bpts[i+1]],
            c=colors[cflip], transform=ccrs.PlateCarree())

    cflip+=1
    cflip = cflip%2
ax2.plot(track3[3, dayend:], track3[2, dayend:],
            c=colors[cflip], transform=ccrs.PlateCarree())

plt.tight_layout()

dashed_blue = lines.Line2D([], [], linestyle="--", dashes=(6, 6), 
    color=cmap_tracks(0), gapcolor=cmap_tracks(0.05), label="Qualification")
dashed_orange= lines.Line2D([], [], linestyle="--", dashes=(6, 6), 
    color=cmap_tracks(0.1), gapcolor=cmap_tracks(0.15), label="COSI")
dashed_green = lines.Line2D([], [], linestyle="--", dashes=(6, 6), 
    color=cmap_tracks(0.2), gapcolor=cmap_tracks(0.25), label="SuperBIT")
fig2.legend(handles=[dashed_blue, dashed_orange, dashed_green],
    frameon=False, fontsize=9, ncol=3, loc="lower center", 
    bbox_to_anchor=(0.5, 0.92))
plt.savefig("trajectory_map.pdf", bbox_inches="tight", dpi=180)
plt.show()