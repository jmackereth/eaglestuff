import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import gridspec
from scipy.stats import gaussian_kde
import numpy as np
import eagle as E
import os
import math
import csv

default_run = "L0050N0752"
default_sim = "/data5/simulations/EAGLE/"+default_run+"/REFERENCE/data"
default_tag = "028_z000p000"

boxsize = E.readAttribute("SUBFIND", default_sim, default_tag, "/Header/BoxSize")
h = E.readAttribute("SUBFIND", default_sim, default_tag, "/Header/HubbleParam")
boxsize = boxsize/h
masstable = E.readAttribute("SUBFIND", default_sim, default_tag, "/Header/MassTable") / h

def ensure_dir(f):
	""" Ensure a a file exists and if not make the relevant path """
	d = os.path.dirname(f)
	if not os.path.exists(d):
			os.makedirs(d)

def correctwrap(rel_pos):
	""" Correct the periodic wrap in EAGLE """
	for i in range(0,len(rel_pos)):
		if abs(rel_pos[i][0]) > (boxsize/2):
			if np.sign(rel_pos[i][0]) == -1:
				rel_pos[i][0] = rel_pos[i][0] + boxsize
			else:
				rel_pos[i][0] = rel_pos[i][0] - boxsize
		if abs(rel_pos[i][1]) > (boxsize/2):
			if np.sign(rel_pos[i][1]) == -1:
				rel_pos[i][1] = rel_pos[i][1] + boxsize
			else:
				rel_pos[i][1] = rel_pos[i][1] - boxsize
		if abs(rel_pos[i][2]) > (boxsize/2):
			if np.sign(rel_pos[i][2]) == -1:
				rel_pos[i][2] = rel_pos[i][2] + boxsize
			else:
				rel_pos[i][2] = rel_pos[i][2] - boxsize
	return np.array(rel_pos)	


def loadparticles(run=default_run,sim=default_sim,tag=default_tag):
	""" This loads the particle data for a given simulation and returns an array with that data. """
	groupnum_type = np.array( [E.readArray("PARTDATA", sim, tag, "/PartType0/GroupNumber"), E.readArray("PARTDATA", sim, tag, "/PartType1/GroupNumber"), E.readArray("PARTDATA", sim, tag, "/PartType4/GroupNumber"), E.readArray("PARTDATA", sim, tag, "/PartType5/GroupNumber")] )
	subgroupnum_type = np.array( [E.readArray("PARTDATA", sim, tag, "/PartType0/SubGroupNumber"), E.readArray("PARTDATA", sim, tag, "/PartType1/SubGroupNumber"), E.readArray("PARTDATA", sim, tag, "/PartType4/SubGroupNumber"), E.readArray("PARTDATA", sim, tag, "/PartType5/SubGroupNumber")] )
	pos_type = np.array( [E.readArray("PARTDATA", sim, tag, "/PartType0/Coordinates"), E.readArray("PARTDATA", sim, tag, "/PartType1/Coordinates"), E.readArray("PARTDATA", sim, tag, "/PartType4/Coordinates"), E.readArray("PARTDATA", sim, tag, "/PartType5/Coordinates")] )
	mass_type = np.array( [E.readArray("PARTDATA", sim, tag, "/PartType0/Mass"), (np.ones(len(pos_type[1]))*masstable[1]) , E.readArray("PARTDATA", sim, tag, "/PartType4/Mass"), E.readArray("PARTDATA", sim, tag, "/PartType5/Mass")])
	vel_type = np.array( [E.readArray("PARTDATA", sim, tag, "/PartType0/Velocity"), E.readArray("PARTDATA", sim, tag, "/PartType1/Velocity"), E.readArray("PARTDATA", sim, tag, "/PartType4/Velocity"), E.readArray("PARTDATA", sim, tag, "/PartType5/Velocity")] )
	stars_abundances = np.array( [E.readArray("PARTDATA", sim, tag, "/PartType4/SmoothedElementAbundance/Hydrogen"), E.readArray("PARTDATA", sim, tag, "/PartType4/SmoothedElementAbundance/Helium"), E.readArray("PARTDATA", sim, tag, "/PartType4/SmoothedElementAbundance/Carbon"), E.readArray("PARTDATA", sim, tag, "/PartType4/SmoothedElementAbundance/Nitrogen"), E.readArray("PARTDATA", sim, tag, "/PartType4/SmoothedElementAbundance/Oxygen"), E.readArray("PARTDATA", sim, tag, "/PartType4/SmoothedElementAbundance/Neon"), E.readArray("PARTDATA", sim, tag, "/PartType4/SmoothedElementAbundance/Magnesium"), E.readArray("PARTDATA", sim, tag, "/PartType4/SmoothedElementAbundance/Silicon"), E.readArray("PARTDATA", sim, tag, "/PartType4/SmoothedElementAbundance/Iron")])
	return np.array([groupnum_type, subgroupnum_type, pos_type, mass_type, vel_type, stars_abundances])
	

def loadfofdat(run=default_run,sim=default_sim,tag=default_tag):
	""" Load Relevant FOF data """
	fsid = np.array(E.readArray("SUBFIND_GROUP", sim, tag, "FOF/FirstSubhaloID"))
	groupnumber = np.array(E.readArray("SUBFIND" , sim, tag, "/Subhalo/GroupNumber"))[fsid]
	CoP = np.array(E.readArray("SUBFIND", sim, tag, "/Subhalo/CentreOfPotential"))[fsid]
	subhalovel = np.array(E.readArray("SUBFIND", sim, tag, "/Subhalo/Velocity"))[fsid]
	r_200 = np.array(E.readArray("SUBFIND_GROUP", sim, tag, "/FOF/Group_R_Crit200"))
	tot_ang_mom = np.array(E.readArray("SUBFIND", sim, tag, "/Subhalo/Stars/Spin"))[fsid]
	stellar_mass = np.array(E.readArray("SUBFIND", sim, tag, "/Subhalo/Stars/Mass") * 1e10)[fsid]
	stellar_abundances = np.array( [ E.readArray("SUBFIND", sim, tag, "/Subhalo/Stars/SmoothedElementAbundance/Hydrogen")[fsid], E.readArray("SUBFIND", sim, tag, "/Subhalo/Stars/SmoothedElementAbundance/Helium")[fsid], E.readArray("SUBFIND", sim, tag, "/Subhalo/Stars/SmoothedElementAbundance/Carbon")[fsid], E.readArray("SUBFIND", sim, tag, "/Subhalo/Stars/SmoothedElementAbundance/Nitrogen")[fsid], E.readArray("SUBFIND", sim, tag, "/Subhalo/Stars/SmoothedElementAbundance/Oxygen")[fsid], E.readArray("SUBFIND", sim, tag, "/Subhalo/Stars/SmoothedElementAbundance/Neon")[fsid], E.readArray("SUBFIND", sim, tag, "/Subhalo/Stars/SmoothedElementAbundance/Magnesium")[fsid], E.readArray("SUBFIND", sim, tag, "/Subhalo/Stars/SmoothedElementAbundance/Silicon")[fsid], E.readArray("SUBFIND", sim, tag, "/Subhalo/Stars/SmoothedElementAbundance/Iron")[fsid]]) 
	return np.array([fsid,groupnumber,CoP,subhalovel,r_200,tot_ang_mom, stellar_mass, stellar_abundances ])

def calctrans(fofdat):
	""" make an array of angular momentum transforms for the central subhalos """
	tot_ang_mom = fofdat[5]
	yaw = np.arccos(np.array(zip(*tot_ang_mom)[1])/(np.sqrt(np.array(zip(*tot_ang_mom)[0])**2+np.array(zip(*tot_ang_mom)[1])**2)))
	pitch = np.arccos(np.array(zip(*tot_ang_mom)[1])/(np.sqrt(np.array(zip(*tot_ang_mom)[1])**2+np.array(zip(*tot_ang_mom)[2])**2)))
	roll = np.arccos(np.array(zip(*tot_ang_mom)[0])/(np.sqrt(np.array(zip(*tot_ang_mom)[0])**2+np.array(zip(*tot_ang_mom)[2])**2)))
	cos = np.cos
	sin = np.sin
	yaw_tran = [np.matrix([[cos(y), -sin(y), 0],[sin(y), cos(y), 0],[0,0,1]]) for y in yaw]
	pitch_tran = [np.matrix([[cos(p), 0, sin(p)],[0,1,0],[-sin(p), 0, cos(p)]]) for p in pitch]
	roll_tran = [np.matrix([[1,0,0],[0,cos(r),-sin(r)],[0,sin(r),cos(r)]]) for r in roll]
	trans = [np.matmul(np.matmul(roll_tran,pitch_tran),yaw_tran) for roll_tran,pitch_tran,yaw_tran in zip(roll_tran,pitch_tran,yaw_tran)]
	return trans


def stackparticles(partdat):
	""" Stack Particles ready for use in halo selection etc """
	parttype0 = np.zeros(len(partdat[0][0]))
	parttype1 = np.ones(len(partdat[0][1]))
	parttype4 = np.ones(len(partdat[0][2]))*4
	parttype5 = np.ones(len(partdat[0][3]))*5
	types = np.hstack((parttype0,parttype1,parttype4,parttype5))
	groupnums = np.hstack((partdat[0][0], partdat[0][1], partdat[0][2], partdat[0][3]))
	subgroupnums = np.hstack((partdat[1][0], partdat[1][1], partdat[1][2], partdat[1][3]))
	positions = np.hstack((partdat[2][0].T, partdat[2][1].T, partdat[2][2].T, partdat[2][3].T))
	#positions = positions.T
	print str(np.shape(positions))
	velocities = np.hstack((partdat[4][0].T, partdat[4][1].T, partdat[4][2].T, partdat[4][3].T))
	#velocities = velocities.T
	masses = np.hstack((partdat[3][0], partdat[3][1], partdat[3][2], partdat[3][3]))
	print str(np.shape(masses))
	partarray = np.dstack((types,groupnums,subgroupnums,positions[0],positions[1], positions[2],velocities[0],velocities[1], velocities[2],masses))
	partarray = partarray[0]
	return partarray

def halo(partstack,fofdat,transform,groupnum, plot=True):
	""" define a central halo using groupnum and see its jz/jc histogram and morphology """
	stack = partstack[(partstack[:,1] == groupnum) & (partstack[:,2] == 0)]
	fofindex = np.where(fofdat[1] == groupnum)
	CoP = fofdat[2][fofindex]
	r200 = fofdat[4][fofindex]
	pos = stack[:,3:6]-(CoP-(boxsize/2))
	#transform = np.array(transform)
	#transform = transform[fofindex]
	subhalovel = fofdat[3][fofindex]
	pos[:,:3] %= boxsize
	pos[:,:3] -= boxsize/2
	
	radii = np.linalg.norm(pos, axis = 1)
	radsort = np.argsort(radii)
	radii = radii[radsort]
	stack = stack[radsort]
	pos = pos[radsort]
	cum_mass = np.cumsum(stack[:,9])
	starinnermass = cum_mass[stack[:,0] == 4]
	starpos = pos[stack[:,0] == 4]
	starradii = np.linalg.norm(starpos, axis = 1)
	radmask = starradii < 0.15*r200
	starvel = stack[:,6:9][stack[:,0] == 4]-subhalovel
	starj = np.array([np.cross(starv,starp) for starv,starp in zip(starvel,starpos)])
	r200j = starj[radmask]
	tot_ang_mom = np.sum(r200j, axis = 1)
	yaw = np.arccos(tot_ang_mom[1]/(np.sqrt(tot_ang_mom[0]**2+tot_ang_mom[1]**2)))
	pitch = np.arccos(tot_ang_mom[1]/(np.sqrt(tot_ang_mom[1]**2+tot_ang_mom[2]**2)))
	roll = np.arccos(tot_ang_mom[0]/(np.sqrt(tot_ang_mom[0]**2+tot_ang_mom[2]**2)))
	cos = np.cos
	sin = np.sin
	yaw_tran = np.matrix([[cos(yaw), -sin(yaw), 0],[sin(yaw), cos(yaw), 0],[0,0,1]])
	pitch_tran = np.matrix([[cos(pitch), 0, sin(pitch)],[0,1,0],[-sin(pitch), 0, cos(pitch)]])
	roll_tran = np.matrix([[1,0,0],[0,cos(roll),-sin(roll)],[0,sin(roll),cos(roll)]])
	transform = np.matmul(np.matmul(roll_tran,pitch_tran),yaw_tran)
	print str(np.shape(transform))
	starpos = np.array([np.matmul(starpos[i],transform) for i in range(0,len(starpos))])[:,0]
	print str(starpos)
	starvel = np.array([np.matmul(starvel[i],transform) for i in range(0,len(starvel))])[:,0]
	starr_xy = np.linalg.norm(np.dstack((starpos[:,0],starpos[:,2]))[0], axis = 1)
	G = 4.302e4
	starv_c = np.sqrt((G*starinnermass)/starr_xy)
	starj = np.array([np.cross(starv,starp) for starv,starp in zip(starvel,starpos)])
	starj_z = starj[:,1]
	starj_c = starv_c*starr_xy
	starjz_jc = (starj_z/starj_c)*10
	print str(starjz_jc)
	
	starmass = stack[:,9][stack[:,0] == 4]
	if plot == True:
		hist, bins = np.histogram(starjz_jc, bins=100, range=(-2,2))
		centers = (bins[:-1] + bins[1:]) / 2
		hist = [float(histo)/float(sum(hist)) for histo in hist]
		xbins = np.arange(0,0.05,0.001)
		ybins = np.arange(-2,2,0.05)
		H, xedges, yedges = np.histogram2d(starjz_jc,starr_xy,bins=(ybins,xbins))
		my_cmap = matplotlib.cm.get_cmap('jet')		my_cmap.set_under('w')
		params = {'axes.labelsize': 14, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'text.usetex': True, 'lines.linewidth' : 2}
		plt.rcParams.update(params)
		fig, ax = plt.subplots(1,2)
		ax[0].plot(centers,hist)
		ax[0].set_ylabel(r'$N/N*$')
		ax[0].set_xlabel(r'$j_z / j_c$')
		ax[0].set_xlim(-3,3)
		ax[1].imshow(H, extent =[0,0.15*r200,-3,3], interpolation='nearest', origin='lower', aspect='auto',cmap=my_cmap, vmin=0.1)
		ax[1].set_xlabel(r'$R$ $[Mpc]$')
		ax[1].set_ylabel(r'$j_z / j_c$')
		plt.show()
	
	#radmask = starradii < 0.15*r200
	#starpos = starpos[radmask]
	#starmass = starmass[radmask]
	
	if plot == True:
		x = starpos[:,0]
		y = starpos[:,2]
		z = starpos[:,1]
		params = {'axes.labelsize': 14, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'text.usetex': True, 'lines.linewidth' : 2}
		plt.rcParams.update(params)
		from mpl_toolkits.mplot3d import Axes3D
		fig = plt.figure()
		my_cmap = matplotlib.cm.get_cmap('jet')		my_cmap.set_under('w')
		ybins = np.arange(-0.06, 0.06, 0.001)
		xbins = np.arange(-0.06, 0.06, 0.001)
		gs = gridspec.GridSpec(2,2)
		ax1 = fig.add_subplot(gs[0,0], projection='3d')
		ax1.scatter(x,y,z, s=starmass)
		ax2 = fig.add_subplot(gs[0,1])
		H, xedges, yedges = np.histogram2d(y,x,bins=(ybins,xbins))
		ax2.imshow(H, extent=[-0.06,0.06,-0.06,0.06], interpolation='nearest', origin='lower', aspect='auto',cmap=my_cmap, vmin=0.1, clim=(0,60))
		#ax2.scatter(x,y, s=mass)
		ax2.set_xlabel(r'$x[Mpc]$')
		ax2.set_ylabel(r'$y[Mpc]$')
		ax3 = fig.add_subplot(gs[1,0])
		H, xedges, yedges = np.histogram2d(z,x,bins=(ybins,xbins))
		ax3.imshow(H, extent=[-0.06,0.06,-0.06,0.06], interpolation='nearest', origin='lower', aspect='auto',cmap=my_cmap, vmin=0.1, clim=(0,60))
		#ax3.scatter(x,z, s=mass)
		ax3.set_xlabel(r'$x[Mpc]$')
		ax3.set_ylabel(r'$z[Mpc]$')
		ax4 = fig.add_subplot(gs[1,1])
		H, xedges, yedges = np.histogram2d(z,y,bins=(ybins,xbins))
		ax4.imshow(H, extent=[-0.06,0.06,-0.06,0.06], interpolation='nearest', origin='lower', aspect='auto',cmap=my_cmap, vmin=0.1, clim=(0,60))
		#ax4.scatter(y,z, s=mass)
		ax4.set_xlabel(r'$y[Mpc]$')
		ax4.set_ylabel(r'$z[Mpc]$')
		fig.subplots_adjust(hspace=0.40, wspace=0.40, right=0.95, top=0.95)
		fof = "FOF "+ str(groupnum)
		fig.text(0.5, 0.96, fof)
		plt.show()



def loadhalos(partdat,fofdat,transform):
	""" finds FOFs with more than 1e6 stellar mass and selects their particles, then sets their position relative to the center of potential and transforms them by the transformation matrix from calctrans. 
		an array is then returned with each halo in the order they appear in fofdat
		TOO SLOW - DO NOT USE """
	mass_cut = np.where(fofdat[6] > 1e6)
	groupnums = fofdat[1][mass_cut]
	CoPs = fofdat[2][mass_cut]
	trans = np.array(transform)[mass_cut]
	subhalovel = fofdat[3][mass_cut]
	solarabunds = [0.706498, 0.280555, 0.00206654, 0.000835626, 0.00549262, 0.00141446, 0.000590706, 0.000682587, 0.00110322]
	solarfeh = np.log10(solarabunds[8]/solarabunds[0])
	solarratios = [solarfeh, np.log10(solarabunds[1]/solarabunds[0])-solarfeh, np.log10(solarabunds[2]/solarabunds[0])-solarfeh, np.log10(solarabunds[3]/solarabunds[0])-solarfeh, np.log10(solarabunds[4]/solarabunds[0])-solarfeh, np.log10(solarabunds[5]/solarabunds[0])-solarfeh, np.log10(solarabunds[6]/solarabunds[0])-solarfeh, np.log10(solarabunds[7]/solarabunds[0])-solarfeh]
	subgroup0, subgroup1, subgroup4, subgroup5 = (partdat[1][0], partdat[1][1], partdat[1][2], partdat[1][3])
	sub0inds0, sub0inds1, sub0inds4, sub0inds5 = (np.where(subgroup0 == 0), np.where(subgroup1 == 0), np.where(subgroup4 == 0), np.where(subgroup5 == 0))
	groupnum0, groupnum1, groupnum4, groupnum5 = (partdat[0][0][sub0inds0], partdat[0][1][sub0inds1], partdat[0][2][sub0inds4], partdat[0][3][sub0inds5])
	pos0, pos1, pos4, pos5 = (partdat[2][0][sub0inds0], partdat[2][1][sub0inds1], partdat[2][2][sub0inds4], partdat[2][3][sub0inds5])
	mass0, mass1, mass4, mass5 = (partdat[3][0][sub0inds0], partdat[3][1][sub0inds1], partdat[3][2][sub0inds4], partdat[3][3][sub0inds5])
	vel0, vel1, vel4, vel5 = (partdat[4][0][sub0inds0], partdat[4][1][sub0inds1], partdat[4][2][sub0inds4], partdat[4][3][sub0inds5])
	abundances = [np.array(partdat[5][i][sub0inds4]) for i in range(0,len(partdat[5]))]
	halos = []
	for i in groupnums:
		fofindex = np.where(groupnums == i)
		CoP = CoPs[fofindex]
		tran = trans[fofindex]
		subvel = subhalovel[fofindex]
		gmask0, gmask1, gmask4, gmask5 = (np.where(groupnum0 == i), np.where(groupnum1 == i), np.where(groupnum4 == i), np.where(groupnum5 == i))
		fofpos0, fofpos1, fofpos4, fofpos5 = (pos0[gmask0], pos1[gmask1], pos4[gmask4], pos5[gmask5])
		relpos0, relpos1, relpos4, relpos5 = (fofpos0-CoP, fofpos1-CoP, fofpos4-CoP, fofpos5-CoP)
		relpos0, relpos1, relpos4, relpos5 = (np.array([np.matmul(rp, tran) for rp in relpos0]), np.array([np.matmul(rp, tran) for rp in relpos1]), np.array([np.matmul(rp, tran) for rp in relpos4]), np.array([np.matmul(rp, tran) for rp in relpos5]))
		fofvel0, fofvel1, fofvel4, fofvel5 = (vel0[gmask0], vel1[gmask1], vel4[gmask4], vel5[gmask5])
		relvel0, relvel1, relvel4, relvel5 = (np.array([bulkvel-subvel for bulkvel in fofvel0]), np.array([bulkvel-subvel for bulkvel in fofvel1]), np.array([bulkvel-subvel for bulkvel in fofvel4]), np.array([bulkvel-subvel for bulkvel in fofvel5]))
		relvel0, relvel1, relvel4, relvel5 = (np.array([np.matmul(rp, tran) for rp in relvel0]), np.array([np.matmul(rp, tran) for rp in relvel1]), np.array([np.matmul(rp, tran) for rp in relvel4]), np.array([np.matmul(rp, tran) for rp in relvel5]))
		fofmass0, fofmass1, fofmass4, fofmass5 = (mass0[gmask0], mass1[gmask1], mass4[gmask4], mass5[gmask5])
		fofabunds = [np.array(abundances)[i][gmask4] for i in range(0, len(partdat[5]))]
		foffeh = np.log10(fofabunds[8]/fofabunds[0])
		fofratios = [foffeh - solarratios[0], (np.log10(fofabunds[1]/fofabunds[0])-foffeh)-solarratios[1], (np.log10(fofabunds[2]/fofabunds[0])-foffeh)-solarratios[2], (np.log10(fofabunds[3]/fofabunds[0])-foffeh)-solarratios[3], (np.log10(fofabunds[4]/fofabunds[0])-foffeh)-solarratios[4], (np.log10(fofabunds[5]/fofabunds[0])-foffeh)-solarratios[5], (np.log10(fofabunds[6]/fofabunds[0])-foffeh)-solarratios[6], (np.log10(fofabunds[7]/fofabunds[0])-foffeh)-solarratios[7]]
		angmom0, angmom1, angmom4, angmom5 = ([np.cross(rpos,v) for rpos,v in zip(relpos0,relvel0)], [np.cross(rpos,v) for rpos,v in zip(relpos1,relvel1)], [np.cross(rpos,v) for rpos,v in zip(relpos4,relvel4)], [np.cross(rpos,v) for rpos,v in zip(relpos5,relvel5)])
		position = np.array( relpos0, relpos1, relpos4, relpos5)
		velocity = np.array( relvel0, relvel1, relvel4, relvel5)
		mass = np.array( fofmass0, fofmass1, fofmass4, fofmass5)
		ang_mom = np.array( angmom0, angmom1, angmom4, angmom5)
		halo = np.array( position, velocity, mass, ang_mom, fofabunds, fofratios)
		halos.append(halo)
		print str(fofindex/len(groupnums))
	return halos


