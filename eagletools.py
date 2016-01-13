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

run = "L0050N0752"
sim = "/data5/simulations/EAGLE/"+run+"/REFERENCE/data"
tag = "028_z000p000"

def ensure_dir(f):
	d = os.path.dirname(f)
	if not os.path.exists(d):
			os.makedirs(d)

def savesubhalo(halo,subgroup,parttype,path="/data5/astjmack/halos/"):
	halo = halo
	subgroup = subgroup
	parttype = parttype
	good_types = [0,4,5]
	boxsize = E.readAttribute("SUBFIND", sim, tag, "/Header/BoxSize")
	h = E.readAttribute("SUBFIND", sim, tag, "/Header/HubbleParam")
	masstable = E.readAttribute("SUBFIND", sim, tag, "/Header/MassTable") / h
	boxsize = boxsize/h
	groupnum = E.readArray("PARTDATA", sim, tag, "/PartType"+str(parttype)+"/GroupNumber")
	subgroupnum = E.readArray("PARTDATA", sim, tag, "/PartType"+str(parttype)+"/SubGroupNumber")
	subgroupnum = subgroupnum[groupnum == halo]
	r_200 = E.readArray("SUBFIND_GROUP", sim, tag, "/FOF/Group_R_Crit200")[halo-1]
	fsid = E.readArray("SUBFIND_GROUP", sim, tag, "FOF/FirstSubhaloID")
	pos = E.readArray("PARTDATA", sim, tag, "/PartType"+str(parttype)+"/Coordinates")[groupnum == halo, :]
	pos = pos[subgroupnum == subgroup, :]
	if parttype in good_types:
		mass = E.readArray("PARTDATA", sim, tag, "/PartType"+str(parttype)+"/Mass")[groupnum == halo]
		mass = mass[subgroupnum == subgroup]
	elif parttype == 1:
		mass = np.ones(len(pos))*masstable[1]
		
	vel = E.readArray("PARTDATA", sim, tag, "/PartType"+str(parttype)+"/Velocity")[groupnum == halo, :]
	vel = vel[subgroupnum == subgroup, :]
	if parttype == 4:
		stars_h = E.readArray("PARTDATA", sim, tag, "/PartType"+str(parttype)+"/SmoothedElementAbundance/Hydrogen")[groupnum == halo]
		stars_fe = E.readArray("PARTDATA", sim, tag, "/PartType"+str(parttype)+"/SmoothedElementAbundance/Iron")[groupnum == halo]
		stars_o = E.readArray("PARTDATA", sim, tag, "/PartType"+str(parttype)+"/SmoothedElementAbundance/Oxygen")[groupnum == halo]
		stars_mg = E.readArray("PARTDATA", sim, tag, "/PartType"+str(parttype)+"/SmoothedElementAbundance/Magnesium")[groupnum == halo]
		starformtime = E.readArray("PARTDATA", sim, tag, "/PartType"+str(parttype)+"/StellarFormationTime")[groupnum == halo]
		stars_h = stars_h[subgroupnum == subgroup]
		stars_fe = stars_fe[subgroupnum == subgroup]
		stars_o = stars_o[subgroupnum == subgroup]
		stars_mg = stars_mg[subgroupnum == subgroup]
		starformtime = starformtime[subgroupnum == subgroup]
		solar_h = 0.706498
		solar_fe = 0.00110322
		solar_mg = 0.000590706
		solar_o = 0.00549262
		solar_fe_h = np.log10(solar_fe/solar_h)
		solar_mg_fe = np.log10(solar_mg/solar_h)-(solar_fe_h)
		solar_o_fe = np.log10(solar_o/solar_h)-(solar_fe_h)
		stars_fe_h = np.log10(stars_fe/stars_h)
		stars_mg_fe = np.log10(stars_mg/stars_h)-(stars_fe_h)
		stars_o_fe = np.log10(stars_o/stars_h)-(stars_fe_h)
		fe_h = np.array([str_fe_h - solar_fe_h for str_fe_h in stars_fe_h])
		mg_fe = np.array([str_a_fe - solar_mg_fe for str_a_fe in stars_mg_fe])
		o_fe = np.array([str_o_fe - solar_o_fe for str_o_fe in stars_o_fe])

	subhaloindex = fsid[halo-1]+subgroup
	CoP = E.readArray("SUBFIND", sim, tag, "/Subhalo/CentreOfPotential")[subhaloindex, :]
	subhalovel = E.readArray("SUBFIND", sim, tag, "/Subhalo/Velocity")[subhaloindex, :]

	#Calculate the abundance ratios (relative to solar abundances from EAGLE)
	

	rel_pos = [[pos[0]-CoP[0],pos[1]-CoP[1],pos[2]-CoP[2]] for pos in pos] #Relative positions

	#re-position overlapped particles
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
	rel_pos = np.array(rel_pos)

	#Make a mask for R_Crit200 and reduce arrays to contain only these values.
	r_crit_mask =[]
	for i in range(0,len(rel_pos)):
		if np.sqrt(rel_pos[i][0]**2+rel_pos[i][1]**2+rel_pos[i][2]**2) <= 0.15*r_200:
			r_crit_mask.append(True)
		else:
			r_crit_mask.append(False)
	r_crit_mask = np.array(r_crit_mask, dtype='bool')
	rel_pos_1 = rel_pos[r_crit_mask]
	mass = mass[r_crit_mask]
	vel = vel[r_crit_mask]
	if parttype == 4:
		fe_h = fe_h[r_crit_mask]
		mg_fe = mg_fe[r_crit_mask]
		o_fe = o_fe[r_crit_mask]
		fe_h = np.array(fe_h)
		mg_fe = np.array(mg_fe)
		o_fe = np.array(o_fe)
	
	"""
	nanmask = np.zeros(len(fe_h))
	for i in range(0, len(fe_h)):
		if (np.isnan(fe_h[i]) == True) | (np.isinf(fe_h[i]) == True) | (np.isnan(mg_fe[i]) == True) | (np.isinf(mg_fe[i]) == True) | (np.isnan(o_fe[i]) == True) | (np.isinf(o_fe[i]) == True):
			nanmask[i] = False
		else:
			nanmask[i] = True

	nanmask = np.array(nanmask, dtype='bool')

	rel_pos_1 = rel_pos_1[nanmask]
	mass = mass[nanmask]
	vel = vel[nanmask]
	fe_h = fe_h[nanmask]
	mg_fe = mg_fe[nanmask]
	o_fe = o_fe[nanmask]
	starformtime = starformtime[nanmask]
	"""
	#Remove galaxy bulk motion from velocities
	vel = [bulkvel-subhalovel for bulkvel in vel]

	#Perform angular momentum calculation
	mv = [m*v for m,v in zip(mass,vel)]	
	ang_mom = [np.cross(rpos,mv) for rpos,mv in zip(rel_pos_1,mv)]
	tot_ang_mom = map(sum, zip(*ang_mom))
	tot_ang_mom = E.readArray("SUBFIND", sim, tag, "/Subhalo/Stars/Spin")[subhaloindex, :]
	print str(tot_ang_mom)
	yaw = np.arccos(tot_ang_mom[1]/(np.sqrt(tot_ang_mom[0]**2+tot_ang_mom[1]**2)))
	pitch = np.arccos(tot_ang_mom[1]/(np.sqrt(tot_ang_mom[1]**2+tot_ang_mom[2]**2)))
	roll = np.arccos(tot_ang_mom[0]/(np.sqrt(tot_ang_mom[0]**2+tot_ang_mom[2]**2)))
	cos = np.cos
	sin = np.sin
	yaw_tran = np.matrix([[cos(yaw), -sin(yaw), 0],[sin(yaw), cos(yaw), 0],[0,0,1]])
	pitch_tran = np.matrix([[cos(pitch), 0, sin(pitch)],[0,1,0],[-sin(pitch), 0, cos(pitch)]])
	roll_tran = np.matrix([[1,0,0],[0,cos(roll),-sin(roll)],[0,sin(roll),cos(roll)]])
	trans = np.array(roll_tran*pitch_tran*yaw_tran)

	#Transform positions and velocities
	r_tran = np.array([np.array([np.dot(i, trans[0]), np.dot(i, trans[1]), np.dot(i,trans[2])]) for i in rel_pos_1])
	vel_tran = np.array([np.array([np.dot(j, trans[0]), np.dot(j, trans[1]), np.dot(j, trans[2])]) for j in vel])

	#Calculate radial position
	R_pos = np.array([np.sqrt(rpos[0]**2 + rpos[2]**2) for rpos in r_tran])
	z_pos = abs(np.array(zip(*r_tran)[1]))
	
	#vertical and Circular angular momentum
	#mv = [m*v for m,v in zip(mass,vel)]
	ang_mom = [np.cross(rpos,v) for rpos,v in zip(r_tran,vel)]

	#Calculate star formation ages
	Mpc = 3.08567758e22
	t_0 = 13.8
	H_0 = h  * 100
	#t_a = [(2*a**(3/2))/(3*H_0)/(1e9*365*24*60*60) for a in starformtime]
	if parttype == 4:
		ages = starformtime #[t_0 - t for t in t_a]
	
	if parttype == 4:
		partarray = np.array([zip(*r_tran)[0], zip(*r_tran)[2], zip(*r_tran)[1], zip(*vel_tran)[0], zip(*vel_tran)[2], zip(*vel_tran)[1], mass, R_pos, z_pos, fe_h, mg_fe, r_200, zip(*ang_mom)[1], ages, o_fe] )
	else:
		partarray = np.array([zip(*r_tran)[0], zip(*r_tran)[2], zip(*r_tran)[1], zip(*vel_tran)[0], zip(*vel_tran)[2], zip(*vel_tran)[1], mass, R_pos, z_pos, r_200, zip(*ang_mom)[1]])
	ensure_dir(path+run+"_FOF"+str(halo)+"_SUB"+str(subgroup)+"/")
	np.save(path+run+"_FOF"+str(halo)+"_SUB"+str(subgroup)+"/part"+str(parttype)+"dat", partarray)

def plothalo(halo,subgroup,parttype, path="/data5/astjmack/halos/"):
	array = np.load(path+run+"_FOF"+str(halo)+"_SUB"+str(subgroup)+"/part"+str(parttype)+"dat.npy")
	x = array[0]
	y = array[1]
	z = array[2]
	mass = array[6]
	params = {'axes.labelsize': 14, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'text.usetex': True, 'lines.linewidth' : 2}
	plt.rcParams.update(params)
	from mpl_toolkits.mplot3d import Axes3D
	fig = plt.figure()
	gs = gridspec.GridSpec(2,2)
	ax1 = fig.add_subplot(gs[0,0], projection='3d')
	ax1.scatter(x,y,z, s=mass)
	ax2 = fig.add_subplot(gs[0,1])
	ax2.scatter(x,y, s=mass)
	ax2.set_xlabel(r'$x[Mpc]$')
	ax2.set_ylabel(r'$y[Mpc]$')
	ax2.set_aspect('equal')
	ax3 = fig.add_subplot(gs[1,0])
	ax3.scatter(x,z, s=mass)
	ax3.set_xlabel(r'$x[Mpc]$')
	ax3.set_ylabel(r'$z[Mpc]$')
	ax3.set_aspect('equal')
	ax4 = fig.add_subplot(gs[1,1])
	ax4.scatter(y,z, s=mass)
	ax4.set_xlabel(r'$y[Mpc]$')
	ax4.set_ylabel(r'$z[Mpc]$')
	ax4.set_aspect('equal')
	fig.subplots_adjust(hspace=0.40, wspace=0.40, right=0.95, top=0.95)
	ensure_dir(path+run+"_FOF"+str(halo)+"_SUB"+str(subgroup)+"/plots/")
	plttitle = path+run+"_FOF"+str(halo)+"_SUB"+str(subgroup)+"/plots/part"+str(parttype)+"pos.png"
	plt.savefig(plttitle, format='png', dpi=1200)

def showhalo(halo,subgroup,parttype, path="/data5/astjmack/halos/"):
	array = np.load(path+run+"_FOF"+str(halo)+"_SUB"+str(subgroup)+"/part"+str(parttype)+"dat.npy")
	x = array[0]
	y = array[1]
	z = array[2]
	mass = array[6]
	params = {'axes.labelsize': 14, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'text.usetex': True, 'lines.linewidth' : 2}
	plt.rcParams.update(params)
	from mpl_toolkits.mplot3d import Axes3D
	fig = plt.figure()
	my_cmap = matplotlib.cm.get_cmap('jet')	my_cmap.set_under('w')
	ybins = np.arange(-0.06, 0.06, 0.001)
	xbins = np.arange(-0.06, 0.06, 0.001)
	gs = gridspec.GridSpec(2,2)
	ax1 = fig.add_subplot(gs[0,0], projection='3d')
	ax1.scatter(x,y,z, s=mass)
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
	fof = "FOF "+ str(halo)
	fig.text(0.5, 0.96, fof)
	ensure_dir(path+run+"_FOF"+str(halo)+"_SUB"+str(subgroup)+"/plots/")
	plttitle = path+run+"_FOF"+str(halo)+"_SUB"+str(subgroup)+"/plots/part"+str(parttype)+"pos.eps"
	plt.show()

def plotalphafe(halo,subgroup,parttype, path="/data5/astjmack/halos/"):
	array = np.load(path+run+"_FOF"+str(halo)+"_SUB"+str(subgroup)+"/part"+str(parttype)+"dat.npy")
	R_pos = array[7]
	z_pos = array[8]
	fe_h = array[9]
	a_fe = array[15]
	r_200 = array[11]
	rad_low = 0.003
	rad_high = 0.015
	z_low = 0.00
	z_high = 0.003
	rad_bins = np.linspace(rad_low,rad_high,num=7, endpoint=True)
	z_bins = np.linspace(z_low,z_high,num=4, endpoint=True)
	#rad_bins = np.linspace(0.0,0.1*r_200,num=7, endpoint=False)
	#z_bins = np.linspace(0.00,0.1*r_200,num=4, endpoint=False)
	#bin data into 6 radial and height bins (as in Hayden et al. 2015)
	radial_fe_h = []
	radial_a_fe = []
	for j in range(0,len(z_bins)-1):
		z_mask = [(np.abs(z_pos) > z_bins[j]) & (np.abs(z_pos) < z_bins[j+1])]
		fe_h_zbin = fe_h[z_mask]
		a_fe_zbin = a_fe[z_mask]
		r_zbin = R_pos[z_mask]
		for i in range(0,len(rad_bins)-1):
			bin_mask = [(np.abs(r_zbin) > rad_bins[i]) & (np.abs(r_zbin) < rad_bins[i+1])]
			fe_h_bin = fe_h_zbin[bin_mask]
			a_fe_bin = a_fe_zbin[bin_mask]
			radial_fe_h.append(fe_h_bin)
			radial_a_fe.append(a_fe_bin)
	xbins = np.linspace(-3, 1, 40)
	ybins = np.linspace(-1.5,1.5, 40)
	my_cmap = matplotlib.cm.get_cmap('jet')	my_cmap.set_under('w')
	#produce plot
	params = {'axes.labelsize': 14, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'text.usetex': True, 'lines.linewidth' : 2, 'axes.titlesize' : 6}
	plt.rcParams.update(params)
	f, ((ax1,ax2,ax3,ax4,ax5,ax6),(ax7,ax8,ax9,ax10,ax11,ax12),(ax13,ax14,ax15,ax16,ax17,ax18)) = plt.subplots(3,6,sharey='row', sharex='row')
	H, xedges, yedges = np.histogram2d(radial_a_fe[0],radial_fe_h[0],bins=(ybins,xbins))
	ax13.imshow(H, extent=[-3,1,-1.5,1.5], interpolation='nearest', origin='lower',aspect=3, cmap=my_cmap, vmin=0.1)
	H, xedges, yedges = np.histogram2d(radial_a_fe[1],radial_fe_h[1],bins=(ybins,xbins))
	ax14.imshow(H, extent=[-3,1,-1.5,1.5], interpolation='nearest', origin='lower',aspect=3, cmap=my_cmap, vmin=0.1)
	H, xedges, yedges = np.histogram2d(radial_a_fe[2],radial_fe_h[2],bins=(ybins,xbins))
	ax15.imshow(H, extent=[-3,1,-1.5,1.5], interpolation='nearest', origin='lower',aspect=3, cmap=my_cmap, vmin=0.1)
	H, xedges, yedges = np.histogram2d(radial_a_fe[3],radial_fe_h[3],bins=(ybins,xbins))
	ax16.imshow(H, extent=[-3,1,-1.5,1.5], interpolation='nearest', origin='lower',aspect=3, cmap=my_cmap, vmin=0.1)
	H, xedges, yedges = np.histogram2d(radial_a_fe[4],radial_fe_h[4],bins=(ybins,xbins))
	ax17.imshow(H, extent=[-3,1,-1.5,1.5], interpolation='nearest', origin='lower',aspect=3, cmap=my_cmap, vmin=0.1)
	H, xedges, yedges = np.histogram2d(radial_a_fe[5],radial_fe_h[5],bins=(ybins,xbins))
	ax18.imshow(H, extent=[-3,1,-1.5,1.5], interpolation='nearest', origin='lower',aspect=3, cmap=my_cmap, vmin=0.1)
	H, xedges, yedges = np.histogram2d(radial_a_fe[6],radial_fe_h[6],bins=(ybins,xbins))
	ax7.imshow(H, extent=[-3,1,-1.5,1.5], interpolation='nearest', origin='lower',aspect=3, cmap=my_cmap, vmin=0.1)
	H, xedges, yedges = np.histogram2d(radial_a_fe[7],radial_fe_h[7],bins=(ybins,xbins))
	ax8.imshow(H, extent=[-3,1,-1.5,1.5], interpolation='nearest', origin='lower',aspect=3, cmap=my_cmap, vmin=0.1)
	H, xedges, yedges = np.histogram2d(radial_a_fe[8],radial_fe_h[8],bins=(ybins,xbins))
	ax9.imshow(H, extent=[-3,1,-1.5,1.5], interpolation='nearest', origin='lower',aspect=3, cmap=my_cmap, vmin=0.1)
	H, xedges, yedges = np.histogram2d(radial_a_fe[9],radial_fe_h[9],bins=(ybins,xbins))
	ax10.imshow(H, extent=[-3,1,-1.5,1.5], interpolation='nearest', origin='lower',aspect=3, cmap=my_cmap, vmin=0.1)
	H, xedges, yedges = np.histogram2d(radial_a_fe[10],radial_fe_h[10],bins=(ybins,xbins))
	ax11.imshow(H, extent=[-3,1,-1.5,1.5], interpolation='nearest', origin='lower',aspect=3, cmap=my_cmap, vmin=0.1)
	H, xedges, yedges = np.histogram2d(radial_a_fe[11],radial_fe_h[11],bins=(ybins,xbins))
	ax12.imshow(H, extent=[-3,1,-1.5,1.5], interpolation='nearest', origin='lower',aspect=3, cmap=my_cmap, vmin=0.1)
	H, xedges, yedges = np.histogram2d(radial_a_fe[12],radial_fe_h[12],bins=(ybins,xbins))
	ax1.imshow(H, extent=[-3,1,-1.5,1.5], interpolation='nearest', origin='lower',aspect=3, cmap=my_cmap, vmin=0.1)
	H, xedges, yedges = np.histogram2d(radial_a_fe[13],radial_fe_h[13],bins=(ybins,xbins))
	ax2.imshow(H, extent=[-3,1,-1.5,1.5], interpolation='nearest', origin='lower',aspect=3, cmap=my_cmap, vmin=0.1)
	H, xedges, yedges = np.histogram2d(radial_a_fe[14],radial_fe_h[14],bins=(ybins,xbins))
	ax3.imshow(H, extent=[-3,1,-1.5,1.5], interpolation='nearest', origin='lower',aspect=3, cmap=my_cmap, vmin=0.1)
	H, xedges, yedges = np.histogram2d(radial_a_fe[15],radial_fe_h[15],bins=(ybins,xbins))
	ax4.imshow(H, extent=[-3,1,-1.5,1.5], interpolation='nearest', origin='lower',aspect=3, cmap=my_cmap, vmin=0.1)	
	H, xedges, yedges = np.histogram2d(radial_a_fe[16],radial_fe_h[16],bins=(ybins,xbins))
	ax5.imshow(H, extent=[-3,1,-1.5,1.5], interpolation='nearest', origin='lower',aspect=3, cmap=my_cmap, vmin=0.1)
	H, xedges, yedges = np.histogram2d(radial_a_fe[17],radial_fe_h[17],bins=(ybins,xbins))
	ax6.imshow(H, extent=[-3,1,-1.5,1.5], interpolation='nearest', origin='lower',aspect=3, cmap=my_cmap, vmin=0.1)
	ax1.set_ylabel(r'$[O/Fe]$')
	ax7.set_ylabel(r'$[O/Fe]$')
	ax13.set_ylabel(r'$[O/Fe]$')
	ax13.set_xlabel(r'$[Fe/H]$')
	ax14.set_xlabel(r'$[Fe/H]$')
	ax15.set_xlabel(r'$[Fe/H]$')
	ax16.set_xlabel(r'$[Fe/H]$')
	ax17.set_xlabel(r'$[Fe/H]$')
	ax18.set_xlabel(r'$[Fe/H]$')
	axes = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18]
	labels = [r'$'+str(rad_bins[0]*1000)+' < R < '+str(rad_bins[1]*1000)+'$ $Kpc$', r'$'+str(rad_bins[1]*1000)+' < R < '+str(rad_bins[2]*1000)+'$ $Kpc$', r'$'+str(rad_bins[2]*1000)+' < R < '+str(rad_bins[3]*1000)+'$ $Kpc$', r'$'+str(rad_bins[3]*1000)+' < R < '+str(rad_bins[4]*1000)+'$ $Kpc$', r'$'+str(rad_bins[4]*1000)+' < R < '+str(rad_bins[5]*1000)+'$ $Kpc$', r'$'+str(rad_bins[5]*1000)+' < R < '+str(rad_bins[6]*1000)+'$ $Kpc$']
	for i in range(0,6):
		axes[i].set_title(labels[i])
		axes[i+6].set_title(labels[i])
		axes[i+12].set_title(labels[i])
	for ax in axes:
		ax.set_ylim([-0.6,1.0])
		ax.set_xlim([-2.5,1.0])
		ax.yaxis.set_ticks(np.arange(-0.6,1.2,0.2))
		ax.xaxis.set_ticks(np.arange(-2.0,2.0,1.0))
	f.subplots_adjust(hspace=0.40, wspace=0, left=0.08, right=0.97, top=0.92, bottom=0.08)
	f.text(0.45, 0.34, r'$'+str(z_bins[0]*1000)+' < |z| < '+str(z_bins[1]*1000)+'$ $Kpc$', fontsize=9)
	f.text(0.45, 0.65, r'$'+str(z_bins[1]*1000)+' < |z| < '+str(z_bins[2]*1000)+'$ $Kpc$', fontsize=9)
	f.text(0.45, 0.96, r'$'+str(z_bins[2]*1000)+' < |z| < '+str(z_bins[3]*1000)+'$ $Kpc$', fontsize=9)
	#f.set_size_inches(12,6, forward=True)
	ax13.legend()

	ensure_dir(path+run+"_FOF"+str(halo)+"_SUB"+str(subgroup)+"/plots/")
	plttitle = path+run+"_FOF"+str(halo)+"_SUB"+str(subgroup)+"/plots/part"+str(parttype)+"alphafehist.png"
	
	plt.savefig(plttitle, format='png', dpi=1200)

def plotmdf(halo,subgroup,parttype, path="/data5/astjmack/halos/"):
	array = np.load(path+run+"_FOF"+str(halo)+"_SUB"+str(subgroup)+"/part"+str(parttype)+"dat.npy")
	R_pos = array[7]
	z_pos = array[8]
	fe_h = array[9]
	a_fe = array[10]
	r_200 = array[11]
	rad_low = 0.003
	rad_high = 0.015
	z_low = 0.00
	z_high = 0.003
	rad_bins = np.linspace(rad_low,rad_high,num=7, endpoint=True)
	z_bins = np.linspace(z_low,z_high,num=4, endpoint=True)
	radial_fe_h = []
	for j in range(0,len(z_bins)-1): 
		z_mask = [(np.abs(z_pos) > z_bins[j]) & (np.abs(z_pos) < z_bins[j+1])]
		fe_h_zbin = fe_h[z_mask]
		r_zbin = R_pos[z_mask]
		for i in range(0,len(rad_bins)-1):
			bin_mask = [(np.abs(r_zbin) > rad_bins[i]) & (np.abs(r_zbin) < rad_bins[i+1])]
			fe_h_bin = fe_h_zbin[bin_mask]
			radial_fe_h.append(fe_h_bin)

	len_zbins = len(radial_fe_h)/len(z_bins)
	fe_h_hists = []
	fe_h_centers = []
	for i in range(0,len(radial_fe_h)):
		hist, bins = np.histogram(radial_fe_h[i], bins=15, range=(-3,1.0))
		rad_feh = radial_fe_h[i]
		centers = (bins[:-1] + bins[1:]) / 2
		hist = [float(histo)/float(sum(hist)) for histo in hist]
		fe_h_hists.append(hist)
		fe_h_centers.append(centers)

	labels = [r'$'+str(rad_bins[0]*1000)+' < R < '+str(rad_bins[1]*1000)+'$ $Kpc$', r'$'+str(rad_bins[1]*1000)+' < R < '+str(rad_bins[2]*1000)+'$ $Kpc$', r'$'+str(rad_bins[2]*1000)+' < R < '+str(rad_bins[3]*1000)+'$ $Kpc$', r'$'+str(rad_bins[3]*1000)+' < R < '+str(rad_bins[4]*1000)+'$ $Kpc$', r'$'+str(rad_bins[4]*1000)+' < R < '+str(rad_bins[5]*1000)+'$ $Kpc$', r'$'+str(rad_bins[5]*1000)+' < R < '+str(rad_bins[6]*1000)+'$ $Kpc$']

	params = {'axes.labelsize': 14, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'text.usetex': True, 'lines.linewidth' : 2}
	plt.rcParams.update(params)
	fig, ax = plt.subplots(3,1, sharex=True)
	
	cmap = colors.LinearSegmentedColormap.from_list('sSFR',['red','green'],256)
	
	for i in range(0,(len(radial_fe_h)/3)):
		ax[2].plot(fe_h_centers[i], fe_h_hists[i], label=labels[i])
		ax[2].set_xlabel(r'$[Fe/H]$')
		ax[2].set_ylabel(r'$N/N*$')
		ax[2].set_title(r'$'+str(z_bins[0]*1000)+' < |z| < '+str(z_bins[1]*1000)+'$ $Kpc$')
		ax[2].legend(loc=2, fontsize='x-small', frameon=False, shadow=False)
	for i in range((len(radial_fe_h)/3),(2*len(radial_fe_h)/3)):
		ax[1].plot(fe_h_centers[i], fe_h_hists[i], label=labels[i-6])
		ax[1].set_ylabel(r'$N/N*$')
		ax[1].set_title(r'$'+str(z_bins[1]*1000)+' < |z| < '+str(z_bins[2]*1000)+'$ $Kpc$')
		ax[1].legend(loc=2, fontsize='x-small', frameon=False, shadow=False)
	for i in range((2*len(radial_fe_h)/3),(len(radial_fe_h))):
		ax[0].plot(fe_h_centers[i], fe_h_hists[i], label=labels[i-12])
		ax[0].set_ylabel(r'$N/N*$')
		ax[0].set_title(r'$'+str(z_bins[2]*1000)+' < |z| < '+str(z_bins[3]*1000)+'$ $Kpc$')
		ax[0].legend(loc=2, fontsize='x-small', frameon=False, shadow=False)
	ensure_dir(path+run+"_FOF"+str(halo)+"_SUB"+str(subgroup)+"/plots/")
	plttitle = path+run+"_FOF"+str(halo)+"_SUB"+str(subgroup)+"/plots/part"+str(parttype)+"mdf.png"
	plt.savefig(plttitle, format='png', dpi=1200)

def selectdiskies(path="/data5/astjmack/halos/"):
	mass = E.readArray("SUBFIND", sim, tag, "/Subhalo/Stars/Mass") * 1e10
	group = E.readArray("SUBFIND", sim, tag, "/Subhalo/GroupNumber")
	subgroup = E.readArray("SUBFIND", sim, tag, "/Subhalo/SubGroupNumber")
	mask = [(mass > 6e10) & (mass < 8e10)]
	mass = mass[mask]
	group = group[mask]
	subgroup = subgroup[mask]
	array = np.array([group, subgroup, mass])
	np.save(path+run+"_massselection.npy", array)
	return array

def plotjpdf(halo,subgroup,parttypec, path="/data5/astjmack/halos/"):
	array_0 = np.load(path+run+"_FOF"+str(halo)+"_SUB"+str(subgroup)+"/part0dat.npy")
	array_1 = np.load(path+run+"_FOF"+str(halo)+"_SUB"+str(subgroup)+"/part1dat.npy")
	array = np.load(path+run+"_FOF"+str(halo)+"_SUB"+str(subgroup)+"/part4dat.npy")
	array_5 = np.load(path+run+"_FOF"+str(halo)+"_SUB"+str(subgroup)+"/part5dat.npy")
	
	gas_r = np.array([np.linalg.norm([x,y,z]) for x,y,z in zip(np.array(array_0[0]),np.array(array_0[1]),np.array(array_0[2]))])
	dm_r = np.array([np.linalg.norm([x,y,z]) for x,y,z in zip(np.array(array_1[0]),np.array(array_1[1]),np.array(array_1[2]))])
	stars_r = np.array([np.linalg.norm([x,y,z]) for x,y,z in zip(np.array(array[0]),np.array(array[1]),np.array(array[2]))])
	bh_r = np.array([np.linalg.norm([x,y,z]) for x,y,z in zip(np.array(array_5[0]),np.array(array_5[1]),np.array(array_5[2]))])
	
	gas_sort = np.argsort(gas_r)
	dm_sort = np.argsort(dm_r)
	stars_sort = np.argsort(stars_r)
	bh_sort = np.argsort(bh_r)
	
	gas_r = gas_r[gas_sort]
	dm_r = dm_r[dm_sort]
	stars_r = stars_r[stars_sort]
	bh_r = bh_r[bh_sort]

	gas_mass = np.array(array_0[6])[gas_sort]
	dm_mass = np.array(array_1[6])[dm_sort]
	stars_mass = np.array(array[6])[stars_sort]
	bh_mass = np.array(array_5[6])[bh_sort]
	
	cum_gas_mass = np.cumsum(gas_mass)
	cum_dm_mass = np.cumsum(dm_mass)
	cum_stars_mass = np.cumsum(stars_mass)
	cum_bh_mass = np.cumsum(bh_mass)
	
	gas_type = np.ones(len(gas_r))*0
	dm_type = np.ones(len(dm_r))
	stars_type = np.ones(len(stars_r))*4
	bh_type = np.ones(len(bh_r))*5

	masses = np.hstack((gas_mass,dm_mass,stars_mass,bh_mass))
	radii = np.hstack((gas_r,dm_r,stars_r,bh_r))
	types = np.hstack((gas_type, dm_type, stars_type, bh_type))

	partarr = np.dstack((masses,radii,types))

	partmass = np.array(zip(*partarr[0])[0])
	partrad = np.array(zip(*partarr[0])[1])
	parttype = np.array(zip(*partarr[0])[2])

	radsort = np.argsort(partrad)
	partmass = partmass[radsort]
	partrad = partrad[radsort]
	parttype = parttype[radsort]

	partind = np.where(parttype == 4)
	cum_mass = np.cumsum(partmass)
	innermasses = cum_mass[partind]

	jz = np.array(array[12])[stars_sort]
	#jc = np.array(array[13])
	R_pos = np.array(array[7])[stars_sort]
	mass = np.array(array[6])
	#v = np.array([array[3], array[4], array[5]])
	#vmag = [np.linalg.norm(vel) for vel in zip(*v)]et.
	G = 4.302e4
	vmag = np.sqrt((G*innermasses)/R_pos)
	jc = (R_pos*vmag)
	jz_jc = (jz/jc)*10
	hist, bins = np.histogram(jz_jc, bins=100, range=(-2,2))
	centers = (bins[:-1] + bins[1:]) / 2
	hist = [float(histo)/float(sum(hist)) for histo in hist]
	xbins = np.arange(0,0.05,0.001)
	ybins = np.arange(-2,2,0.05)
	H, xedges, yedges = np.histogram2d(jz_jc,R_pos,bins=(ybins,xbins))
	my_cmap = matplotlib.cm.get_cmap('jet')	my_cmap.set_under('w')
	params = {'axes.labelsize': 14, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'text.usetex': True, 'lines.linewidth' : 2}
	plt.rcParams.update(params)
	fig, ax = plt.subplots(1,2)
	ax[0].plot(centers,hist)
	ax[0].set_ylabel(r'$P(N)$')
	ax[0].set_xlabel(r'$j_z / j_c$')
	ax[0].set_xlim(-3,3)
	#ax[1].scatter(R_pos, jz_jc)
	ax[1].imshow(H, extent =[0,max(R_pos),-3,3], interpolation='nearest', origin='lower', aspect='auto',cmap=my_cmap, vmin=0.1)
	ax[1].set_xlabel(r'$R$ $[Mpc]$')
	ax[1].set_ylabel(r'$j_z / j_c$')
	plttitle = path+run+"_FOF"+str(halo)+"_SUB"+str(subgroup)+"/plots/part"+str(parttypec)+"jpdf.png"
	plt.savefig(plttitle, format='png', dpi=1200)
	plt.show()


#def combinehalos(halolist,parttype, path="/data5/astjmack/halos/"):
	

def plotfeage(halo,subgroup,parttype, path="/data5/astjmack/halos/"):
	array = np.load(path+run+"_FOF"+str(halo)+"_SUB"+str(subgroup)+"/part"+str(parttype)+"dat.npy")
	feh = array[9]
	age = array[14]
	print str(np.shape(age))
	params = {'axes.labelsize': 14, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'text.usetex': True, 'lines.linewidth' : 2}
	plt.rcParams.update(params)
	fig, ax = plt.subplots(1,1)
	ax.scatter(age,feh)
	plt.show()

def plotmassmetallicity(path="/data5/astjmack/halos/"):
	spin = E.readArray("SUBFIND", sim, tag, "/Subhalo/Stars/Spin")
	return spin

def toomdiag(halo,subgroup,parttype, path="/data5/astjmack/halos/"):
	array = np.load(path+run+"_FOF"+str(halo)+"_SUB"+str(subgroup)+"/part"+str(parttype)+"dat.npy")
	x = np.array(array[0])
	y = np.array(array[1])
	z = np.array(array[2])
	v_x = np.array(array[3])
	v_y = np.array(array[4])
	v_z = np.array(array[5])
	zmask = z < 0.001
	x = x[zmask]
	y = y[zmask]
	z = z[zmask]
	v_x = v_x[zmask]
	v_y = v_y[zmask]
	v_z = v_z[zmask]
	v_mag = (v_x**2 + v_y**2)**0.5
	r = (x**2 + y**2)**0.5
	fig, ax = plt.subplots(1,1)
	ax.scatter(r, v_mag)
	plt.show()
 

#xy = np.vstack([radial_fe_h[0],radial_a_fe[0]])
	#z = gaussian_kde(xy)(xy)
	#ax13.scatter(radial_fe_h[0],radial_a_fe[0], c=z, edgecolors='none')
	#xy = np.vstack([radial_fe_h[1],radial_a_fe[1]])
	#z = gaussian_kde(xy)(xy)
	#ax14.scatter(radial_fe_h[1],radial_a_fe[1], c=z, edgecolors='none')
	#xy = np.vstack([radial_fe_h[2],radial_a_fe[2]])
	#z = gaussian_kde(xy)(xy)
	#ax15.scatter(radial_fe_h[2],radial_a_fe[2], c=z, edgecolors='none')
	#xy = np.vstack([radial_fe_h[3],radial_a_fe[3]])
	#z = gaussian_kde(xy)(xy)
	#ax16.scatter(radial_fe_h[3],radial_a_fe[3],  c=z, edgecolors='none')
	#xy = np.vstack([radial_fe_h[4],radial_a_fe[4]])
	#z = gaussian_kde(xy)(xy)
	#ax17.scatter(radial_fe_h[4],radial_a_fe[4],  c=z, edgecolors='none')
	#xy = np.vstack([radial_fe_h[5],radial_a_fe[5]])
	#z = gaussian_kde(xy)(xy)
	#ax18.scatter(radial_fe_h[5],radial_a_fe[5],  c=z, edgecolors='none')
	#xy = np.vstack([radial_fe_h[6],radial_a_fe[6]])
	#z = gaussian_kde(xy)(xy)
	#ax7.scatter(radial_fe_h[6],radial_a_fe[6],  c=z, edgecolors='none')
	#xy = np.vstack([radial_fe_h[7],radial_a_fe[7]])
	#z = gaussian_kde(xy)(xy)
	#ax8.scatter(radial_fe_h[7],radial_a_fe[7],  c=z, edgecolors='none')
	#xy = np.vstack([radial_fe_h[8],radial_a_fe[8]])
	#z = gaussian_kde(xy)(xy)
	#ax9.scatter(radial_fe_h[8],radial_a_fe[8],  c=z, edgecolors='none')
	#xy = np.vstack([radial_fe_h[9],radial_a_fe[9]])
	#z = gaussian_kde(xy)(xy)
	#ax10.scatter(radial_fe_h[9],radial_a_fe[9],  c=z, edgecolors='none')
	#xy = np.vstack([radial_fe_h[10],radial_a_fe[10]])
	#z = gaussian_kde(xy)(xy)
	#ax11.scatter(radial_fe_h[10],radial_a_fe[10],  c=z, edgecolors='none')
	#xy = np.vstack([radial_fe_h[11],radial_a_fe[11]])
	#z = gaussian_kde(xy)(xy)
	#ax12.scatter(radial_fe_h[11],radial_a_fe[11],  c=z, edgecolors='none')
	#xy = np.vstack([radial_fe_h[12],radial_a_fe[12]])
	#z = gaussian_kde(xy)(xy)
	#ax1.scatter(radial_fe_h[12],radial_a_fe[12],  c=z, edgecolors='none')
	#xy = np.vstack([radial_fe_h[13],radial_a_fe[13]])
	#z = gaussian_kde(xy)(xy)
	#ax2.scatter(radial_fe_h[13],radial_a_fe[13],  c=z, edgecolors='none')
	#xy = np.vstack([radial_fe_h[14],radial_a_fe[14]])
	#z = gaussian_kde(xy)(xy)
	#ax3.scatter(radial_fe_h[14],radial_a_fe[14],  c=z, edgecolors='none')
	#xy = np.vstack([radial_fe_h[15],radial_a_fe[15]])
	#z = gaussian_kde(xy)(xy)
	#ax4.scatter(radial_fe_h[15],radial_a_fe[15],  c=z, edgecolors='none')
	#xy = np.vstack([radial_fe_h[16],radial_a_fe[16]])
	#z = gaussian_kde(xy)(xy)
	#ax5.scatter(radial_fe_h[16],radial_a_fe[16],  c=z, edgecolors='none')
	#xy = np.vstack([radial_fe_h[17],radial_a_fe[17]])
	#z = gaussian_kde(xy)(xy)
	#ax6.scatter(radial_fe_h[17],radial_a_fe[17],  c=z, edgecolors='none')

