#plot MM elements

import transfer_matrix as tm
import numpy as np
import matplotlib.pyplot as plt


def hwp_band5(center_nu):
    saph_w = 300./(2.0*0.317*center_nu)
    ratio = saph_w/3.75#Rescale AR thickness from the 95/150 AHWP band

    #Thickness
    thicks = np.array([0.5*ratio, 0.31*ratio, 0.257*ratio, 
                       saph_w, saph_w, saph_w, saph_w, saph_w,
                       0.257*ratio, 0.31*ratio, 0.5*ratio])
    #Indices: five saph layers sandwiched between 3 AR layers
    idxs = np.zeros((11,2))
    idxs[0] = [1.268,1.268]
    idxs[1] = [1.979, 1.979]
    idxs[2] = [2.855, 2.855]
    idxs[3:8] = [3.019, 3.336]
    idxs[8] = idxs[2]
    idxs[9] = idxs[1]
    idxs[10] = idxs[0]
    #Losses: dielectric constant along two axes
    losses = np.ones((11,2))*1.2e-3
    losses[3:8] = [2.3e-4, 1.25e-4]
    #Angles: rotation of the saph layers for birefringence
    angles = np.array([0.,0.,0., 22.9,-50.,0.,50.,-22.9 ,0.,0.,0.])*np.pi/180.0

    return thicks, idxs, losses, angles

def hwp_band3(center_nu): 
    saph_w = 300./(2.0*0.317*center_nu)
    ratio = saph_w/3.75#Rescale AR thickness from the 95/150 AHWP band

    #Thickness
    thicks = np.array([0.5*ratio, 0.31*ratio, 0.257*ratio, 
                        saph_w, saph_w, saph_w,
                       0.257*ratio, 0.31*ratio, 0.5*ratio])
    #Indices: five saph layers sandwiched between 3 AR layers
    idxs = np.zeros((9,2))
    idxs[0] = [1.268,1.268]
    idxs[1] = [1.979, 1.979]
    idxs[2] = [2.855, 2.855]
    idxs[3:6] = [3.019, 3.336]
    idxs[6] = idxs[2]
    idxs[7] = idxs[1]
    idxs[8] = idxs[0]
    #Losses: dielectric constant along two axes
    losses = np.ones((9,2))*1.2e-3
    losses[3:6] = [2.3e-4, 1.25e-4]
    #Angles: rotation of the saph layers for birefringence
    angles = np.array([0.,0.,0., 0.,54.0, 0., 0.,0.,0.])*np.pi/180.0

    return thicks, idxs, losses, angles

def hwp_band(center_nu):
    saph_w = 300./(2.0*0.317*center_nu)
    ratio = saph_w/3.75#Rescale AR thickness from the 95/150 AHWP band

    #Thickness
    thicks = np.array([0.5*ratio, 0.31*ratio, 0.257*ratio, saph_w,
                               0.257*ratio, 0.31*ratio, 0.5*ratio])
    #Indices: five saph layers sandwiched between 3 AR layers
    idxs = np.zeros((7,2))
    idxs[0] = [1.268,1.268]
    idxs[1] = [1.979, 1.979]
    idxs[2] = [2.855, 2.855]
    idxs[3] = [3.019, 3.336]
    idxs[4] = idxs[2]
    idxs[5] = idxs[1]
    idxs[6] = idxs[0]
    #Losses: dielectric constant along two axes
    losses = np.ones((7,2))*1.2e-3
    losses[3] = [2.3e-4, 1.25e-4]
    #Angles: rotation of the saph layers for birefringence
    angles = np.zeros(7)

    return thicks, idxs, losses, angles

def stack_builder(thicknesses, indices, losses, angles):
    """
    Creates a stack of materials, as defined in transfer_matrix.py
    Arguments:
    ------------
    thicknesses : (N,1) array of floats 
        thickness of each HWP layer in mm
    indices     : (N,2) array of floats 
        ordinary and extraordinary indices for each layer
    losses      : (N,2) array of floats 
        loss tangents in each layer.
    angles      : (N,1) array of floats 
        angle between (extraordinary) axis and stack axis for each layer, in radians
    """

    if (thicknesses.size != angles.size or 2 * thicknesses.size != indices.size
        or 2*thicknesses.size != losses.size):
        raise ValueError("There is a mismatch in the sizes of the inputs for the HWP stack")

    # Make a list of materials, with a name that corresponds to their position in the stack
    material_stack=[]
    for i in range(thicknesses.size):

        if (indices[i,0]==indices[i,1] and losses[i,0]==losses[i,1]):
            isotro_str = "isotropic"
        else:
            isotro_str = "uniaxial"

        material_stack.append(tm.material(indices[i,0], indices[i,1],
            losses[i,0], losses[i,1], str(i), materialType=isotro_str))

    stack = tm.Stack(thicknesses*tm.mm, material_stack, angles)
    return stack 

def compute_mueller(stack, freq, vartheta):
    """
    Compute the unrotated Mueller Matrix

    Arguments
    -------
    freq : float
        Frequency in GHz
    vartheta : float
        Incidence angle on HWP in radians
    """
    return(tm.Mueller(stack, frequency=1.0e9*freq, incidenceAngle=vartheta,
        rotation=0., reflected=False))

def rotate_hwp_mueller(M, theta):
	cst = np.cos(np.radians(2*theta))
	sst = np.sin(np.radians(2*theta))
	rp = np.array([[1., 0, 0, 0], [0, cst, sst, 0], [0, -sst, cst, 0], [0,0,0,1]])
	rm = np.array([[1., 0, 0, 0], [0, cst, -sst, 0], [0, sst, cst, 0], [0,0,0,1]])
	return rm@M@rp

nfsamp = 1401
t,i,l,a = hwp_band(185)
stack1 = stack_builder(t,i,l,a)
t,i,l,a = hwp_band3(185)
stack3 = stack_builder(t,i,l,a)
t,i,l,a = hwp_band5(185)
stack5 = stack_builder(t,i,l,a)
mm1 = np.zeros((nfsamp, 4,4))
mm3 = np.zeros((nfsamp, 4,4))
mm5 = np.zeros((nfsamp, 4,4))
freq_range = np.linspace(120, 260, nfsamp)

for i, freq in enumerate(freq_range):
	mm1[i] = compute_mueller(stack1, freq, 0.)
	mm3[i] = rotate_hwp_mueller(compute_mueller(stack3, freq, 0.), -32.21)
	mm5[i] = compute_mueller(stack5, freq, 0.)

#Transmission parameters
t0i1 = 0.5*mm1[:,0,0]
t0i3 = 0.5*mm3[:,0,0]
t0i5 = 0.5*mm5[:,0,0]

t0q1 = 0.25*(mm1[:,1,1]+mm1[:,2,2])
t0q3 = 0.25*(mm3[:,1,1]+mm3[:,2,2])
t0q5 = 0.25*(mm5[:,1,1]+mm5[:,2,2])

t0u1 = 0.25*(mm1[:,1,2]-mm1[:,2,1])
t0u3 = 0.25*(mm3[:,1,2]-mm3[:,2,1])
t0u5 = 0.25*(mm5[:,1,2]-mm5[:,2,1])

t41 = 0.25*np.sqrt( (mm1[:,1,1]-mm1[:,2,2])**2 +(mm1[:,1,2]+mm1[:,2,1])**2 )
t43 = 0.25*np.sqrt( (mm3[:,1,1]-mm3[:,2,2])**2 +(mm3[:,1,2]+mm3[:,2,1])**2 )
t45 = 0.25*np.sqrt( (mm5[:,1,1]-mm5[:,2,2])**2 +(mm5[:,1,2]+mm5[:,2,1])**2 )

phi41 = 0.25*np.degrees(np.arctan2(mm1[:,1,2]+mm1[:,2,1], mm1[:,1,1]-mm1[:,2,2]))
phi43 = 0.25*np.degrees(np.arctan2(mm3[:,1,2]+mm3[:,2,1], mm3[:,1,1]-mm3[:,2,2]))
phi45 = 0.25*np.degrees(np.arctan2(mm5[:,1,2]+mm5[:,2,1], mm5[:,1,1]-mm5[:,2,2]))

epsi1 = t41/(t0i1+t0q1)
epsi3 = t43/(t0i3+t0q3)
epsi5 = t45/(t0i5+t0q5)

f1, ax1 = plt.subplots(2, figsize=(3.5, 3.5), sharex=True)
ax1[0].plot(freq_range, epsi1, label="1BR")
ax1[0].plot(freq_range, epsi3, label="3BR")
ax1[0].plot(freq_range, epsi5, label="5BR")
ax1[0].fill_betweenx([0,1.1], 130, 170, color="tab:grey", alpha=0.5)
ax1[0].fill_betweenx([0,1.1], 190, 250, color="tab:grey", alpha=0.5)
ax1[0].set_ylabel("Polarisation efficiency", fontsize=9)
ax1[0].set_ylim(0.5, 1.05)
ax1[0].set_xlim(120, 260)
ax1[0].tick_params(labelsize=9)

ax1[1].plot(freq_range, phi41)
ax1[1].plot(freq_range, phi43)
ax1[1].plot(freq_range, phi45)
ax1[1].set_ylabel("Phase shift (°)", fontsize=9)
ax1[1].set_xlabel("Frequency (GHz)", fontsize=9)
ax1[1].fill_betweenx([-10,5], 130, 170, color="tab:grey", alpha=0.5)
ax1[1].fill_betweenx([-10,5], 190, 250, color="tab:grey", alpha=0.5)
ax1[1].set_ylim(-8,5)
ax1[1].tick_params(labelsize=9)
f1.tight_layout()
f1.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.53, 0.5), fontsize=9)
plt.savefig("hwp_fom.pdf", bbox_inches="tight")
f2, ax2 = plt.subplots(4,4, figsize=(7,7), sharex=True, sharey=True)
mideal = np.eye(4)
mideal[2,2] = -1
mideal[3,3] = -1
stokes_str = ["I", "Q", "U", "V"]
for i, ax in enumerate(ax2.flat):
	elem_str = stokes_str[int(i/4)]+stokes_str[i%4]

	if i==0:
		l1 = ax.plot(freq_range, mm1[:, int(i/4), i%4]-mideal[int(i/4), i%4], label="1BR")
		l2 = ax.plot(freq_range, mm3[:, int(i/4), i%4]-mideal[int(i/4), i%4], label="3BR")
		l3 = ax.plot(freq_range, mm5[:, int(i/4), i%4]-mideal[int(i/4), i%4], label="5BR")
	else:
		ax.plot(freq_range, mm1[:, int(i/4), i%4]-mideal[int(i/4), i%4])
		ax.plot(freq_range, mm3[:, int(i/4), i%4]-mideal[int(i/4), i%4])
		ax.plot(freq_range, mm5[:, int(i/4), i%4]-mideal[int(i/4), i%4])

	ax.fill_betweenx([-0.5,0.5], 130, 170, color="tab:grey", alpha=0.5)
	ax.fill_betweenx([-0.5,0.5], 190, 250, color="tab:grey", alpha=0.5)
	ax.set_ylim(-0.5, 0.5)
	ax.set_xlim(120, 260)
	ax.text(170, 0.4, fr"$M_{{{elem_str}}}$", fontsize=9)
	ax.tick_params(direction="in", labelsize=9)
f2.supxlabel("Frequency (GHz)", fontsize=9, x=0.55, y=0.02)
f2.supylabel("Deviation from ideal HWP", fontsize=9, y=0.53, x=0.04)
f2.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.54, 0.96), fontsize=9)
f2.tight_layout(w_pad=0.2, h_pad=-0.3)
plt.savefig("hwp_mueller_matrix.pdf", bbox_inches="tight")
plt.show()
