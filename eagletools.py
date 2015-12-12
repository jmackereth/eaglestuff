import matplotlib.pyplot as plt
from matplotlib import colors
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
	boxsize = E.readAttribute("SUBFIND", sim, tag, "/Header/BoxSize")
	h = E.readAttribute("SUBFIND", sim, tag, "/Header/HubbleParam")
	boxsize = boxsize/h
	groupnum = E.readArray("PARTDATA", sim, tag, "/PartType"+str(parttype)+"/GroupNumber")
	subgroupnum = E.readArray("PARTDATA", sim, tag, "/PartType"+str(parttype)+"/SubGroupNumber")
	r_200 = E.readArray("SUBFIND_GROUP", sim, tag, "/FOF/Group_R_Crit200")[halo-1]
	fsid = E.readArray("SUBFIND_GROUP", sim, tag, "FOF/FirstSubhaloID")
	pos = E.readArray("PARTDATA", sim, tag, "/PartType"+str(parttype)+"/Coordinates")[groupnum == halo, :]
	mass = E.readArray("PARTDATA", sim, tag, "/PartType"+str(parttype)+"/Mass")[groupnum == halo]
	vel = E.readArray("PARTDATA", sim, tag, "/PartType"+str(parttype)+"/Velocity")[groupnum == halo, :]
	stars_h = E.readArray("PARTDATA", sim, tag, "/PartType"+str(parttype)+"/SmoothedElementAbundance/Hydrogen")[groupnum == halo]
	stars_fe = E.readArray("PARTDATA", sim, tag, "/PartType"+str(parttype)+"/SmoothedElementAbundance/Iron")[groupnum == halo]
	stars_o = E.readArray("PARTDATA", sim, tag, "/PartType"+str(parttype)+"/SmoothedElementAbundance/Oxygen")[groupnum == halo]
	stars_mg = E.readArray("PARTDATA", sim, tag, "/PartType"+str(parttype)+"/SmoothedElementAbundance/Magnesium")[groupnum == halo]
	subgroupnum = subgroupnum[groupnum == halo]
	#subset = list(set(subgroupnum))
	#subset = np.array(subset)
	#subsetsort = np.argsort(subset)
	#subset = subset[subsetsort]
	#subset = subset[subgroup]
	subset = subgroup
	pos = pos[subgroupnum == subset, :]
	mass = mass[subgroupnum == subset]
	vel = vel[subgroupnum == subset, :]
	stars_h = stars_h[subgroupnum == subset]
	stars_fe = stars_fe[subgroupnum == subset]
	stars_o = stars_o[subgroupnum == subset]
	stars_mg = stars_mg[subgroupnum == subset]
	subhaloindex = fsid[halo-1]+subgroup
	CoP = E.readArray("SUBFIND", sim, tag, "/Subhalo/CentreOfPotential")[subhaloindex, :]
	subhalovel = E.readArray("SUBFIND", sim, tag, "/Subhalo/Velocity")[subhaloindex, :]

	#Calculate the abundance ratios (relative to solar abundances from EAGLE)
	solar_h = 0.706498
	solar_fe = 0.00110322
	solar_mg = 0.000590706
	solar_fe_h = np.log10(solar_fe/solar_h)
	solar_a_fe = np.log10(solar_mg/solar_h)-(solar_fe_h)
	stars_fe_h = np.log10(stars_fe/stars_h)
	stars_a_fe = np.log10(stars_mg/stars_h)-(stars_fe_h)
	fe_h = np.array([str_fe_h - solar_fe_h for str_fe_h in stars_fe_h])
	a_fe = np.array([str_a_fe - solar_a_fe for str_a_fe in stars_a_fe])

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
	fe_h = fe_h[r_crit_mask]
	a_fe = a_fe[r_crit_mask]
	fe_h = np.array(fe_h)
	a_fe = np.array(a_fe)

	nanmask = np.zeros(len(fe_h))
	for i in range(0, len(fe_h)):
		if (np.isnan(fe_h[i]) == True) | (np.isinf(fe_h[i]) == True) | (np.isnan(a_fe[i]) == True) | (np.isinf(a_fe[i]) == True):
			nanmask[i] = False
		else:
			nanmask[i] = True

	nanmask = np.array(nanmask, dtype='bool')

	rel_pos_1 = rel_pos_1[nanmask]
	mass = mass[nanmask]
	vel = vel[nanmask]
	fe_h = fe_h[nanmask]
	a_fe = a_fe[nanmask]

	#Remove galaxy bulk motion from velocities
	vel = [bulkvel-subhalovel for bulkvel in vel]

	#Perform angular momentum calculation
	mv = [m*v for m,v in zip(mass,vel)]	
	ang_mom = [np.cross(rpos,mv) for rpos,mv in zip(rel_pos_1,mv)]
	tot_ang_mom = map(sum, zip(*ang_mom))
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
	R_pos = np.array([np.sqrt(rpos[0]**2 + rpos[1]**2) for rpos in r_tran])
	z_pos = abs(np.array(zip(*r_tran)[2]))
	
	partarray = np.array([zip(*r_tran)[0], zip(*r_tran)[1], zip(*r_tran)[2], zip(*vel_tran)[0], zip(*vel_tran)[1], zip(*vel_tran)[2], mass, R_pos, z_pos, fe_h, a_fe, r_200])
	ensure_dir(path+run+"_FOF"+str(halo)+"_SUB"+str(subgroup)+"/")
	np.save(path+run+"_FOF"+str(halo)+"_SUB"+str(subgroup)+"/part"+str(parttype)+"dat", partarray)

def plothalo(datfile,halo,subgroup,parttype, path="/data5/astjmack/halos/"):
	array = np.load(datfile)
	x = array[0]
	y = array[1]
	z = array[2]
	mass = array[6]
	from mpl_toolkits.mplot3d import Axes3D
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(x,y,z, s=mass, edgecolor='none')
	ensure_dir(path+run+"_FOF"+str(halo)+"_SUB"+str(subgroup)+"/plots/")
	plttitle = path+run+"_FOF"+str(halo)+"_SUB"+str(subgroup)+"/plots/part"+str(parttype)+"pos.eps"
	plt.savefig(plttitle, format='eps', dpi=1200)

def plotalphafe(datfile,halo,subgroup,parttype, path="/data5/astjmack/halos/"):
	array = np.load(datfile)
	R_pos = array[7]
	z_pos = array[8]
	fe_h = array[9]
	a_fe = array[10]
	r_200 = array[11]
	rad_bins = np.linspace(float(min(R_pos)),0.15*r_200,num=7, endpoint=False)
	z_bins = np.linspace(0.0,0.08*r_200,num=4, endpoint=False)

	#bin data into 6 radial and height bins (as in Hayden et al. 2015)
	radial_fe_h = []
	radial_a_fe = []
	for j in range(0,len(z_bins)-1):
		z_mask = [(np.abs(z_pos) > z_bins[j]) & (np.abs(z_pos) < z_bins[j+1])]
		fe_h_zbin = fe_h[z_mask]
		a_fe_zbin = a_fe[z_mask]
		r_zbin = R_pos[z_mask]
		for i in range(0,len(rad_bins)-1):
			bin_mask = [(r_zbin > rad_bins[i]) & (r_zbin < rad_bins[i+1])]
			fe_h_bin = fe_h_zbin[bin_mask]
			a_fe_bin = a_fe_zbin[bin_mask]
			radial_fe_h.append(fe_h_bin)
			radial_a_fe.append(a_fe_bin)

	#produce plot
	params = {'axes.labelsize': 14, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'text.usetex': True, 'lines.linewidth' : 2}
	plt.rcParams.update(params)
	f, ((ax1,ax2,ax3,ax4,ax5,ax6),(ax7,ax8,ax9,ax10,ax11,ax12),(ax13,ax14,ax15,ax16,ax17,ax18)) = plt.subplots(3,6,sharey='row', sharex='row')
	xy = np.vstack([radial_fe_h[0],radial_a_fe[0]])
	z = gaussian_kde(xy)(xy)
	ax13.scatter(radial_fe_h[0],radial_a_fe[0], c=z, edgecolors='none')
	xy = np.vstack([radial_fe_h[1],radial_a_fe[1]])
	z = gaussian_kde(xy)(xy)
	ax14.scatter(radial_fe_h[1],radial_a_fe[1], c=z, edgecolors='none')
	xy = np.vstack([radial_fe_h[2],radial_a_fe[2]])
	z = gaussian_kde(xy)(xy)
	ax15.scatter(radial_fe_h[2],radial_a_fe[2], c=z, edgecolors='none')
	xy = np.vstack([radial_fe_h[3],radial_a_fe[3]])
	z = gaussian_kde(xy)(xy)
	ax16.scatter(radial_fe_h[3],radial_a_fe[3],  c=z, edgecolors='none')
	xy = np.vstack([radial_fe_h[4],radial_a_fe[4]])
	z = gaussian_kde(xy)(xy)
	ax17.scatter(radial_fe_h[4],radial_a_fe[4],  c=z, edgecolors='none')
	xy = np.vstack([radial_fe_h[5],radial_a_fe[5]])
	z = gaussian_kde(xy)(xy)
	ax18.scatter(radial_fe_h[5],radial_a_fe[5],  c=z, edgecolors='none')
	xy = np.vstack([radial_fe_h[6],radial_a_fe[6]])
	z = gaussian_kde(xy)(xy)
	ax7.scatter(radial_fe_h[6],radial_a_fe[6],  c=z, edgecolors='none')
	xy = np.vstack([radial_fe_h[7],radial_a_fe[7]])
	z = gaussian_kde(xy)(xy)
	ax8.scatter(radial_fe_h[7],radial_a_fe[7],  c=z, edgecolors='none')
	xy = np.vstack([radial_fe_h[8],radial_a_fe[8]])
	z = gaussian_kde(xy)(xy)
	ax9.scatter(radial_fe_h[8],radial_a_fe[8],  c=z, edgecolors='none')
	xy = np.vstack([radial_fe_h[9],radial_a_fe[9]])
	z = gaussian_kde(xy)(xy)
	ax10.scatter(radial_fe_h[9],radial_a_fe[9],  c=z, edgecolors='none')
	xy = np.vstack([radial_fe_h[10],radial_a_fe[10]])
	z = gaussian_kde(xy)(xy)
	ax11.scatter(radial_fe_h[10],radial_a_fe[10],  c=z, edgecolors='none')
	xy = np.vstack([radial_fe_h[11],radial_a_fe[11]])
	z = gaussian_kde(xy)(xy)
	ax12.scatter(radial_fe_h[11],radial_a_fe[11],  c=z, edgecolors='none')
	xy = np.vstack([radial_fe_h[12],radial_a_fe[12]])
	z = gaussian_kde(xy)(xy)
	ax1.scatter(radial_fe_h[12],radial_a_fe[12],  c=z, edgecolors='none')
	xy = np.vstack([radial_fe_h[13],radial_a_fe[13]])
	z = gaussian_kde(xy)(xy)
	ax2.scatter(radial_fe_h[13],radial_a_fe[13],  c=z, edgecolors='none')
	xy = np.vstack([radial_fe_h[14],radial_a_fe[14]])
	z = gaussian_kde(xy)(xy)
	ax3.scatter(radial_fe_h[14],radial_a_fe[14],  c=z, edgecolors='none')
	xy = np.vstack([radial_fe_h[15],radial_a_fe[15]])
	z = gaussian_kde(xy)(xy)
	ax4.scatter(radial_fe_h[15],radial_a_fe[15],  c=z, edgecolors='none')
	xy = np.vstack([radial_fe_h[16],radial_a_fe[16]])
	z = gaussian_kde(xy)(xy)
	ax5.scatter(radial_fe_h[16],radial_a_fe[16],  c=z, edgecolors='none')
	xy = np.vstack([radial_fe_h[17],radial_a_fe[17]])
	z = gaussian_kde(xy)(xy)
	ax6.scatter(radial_fe_h[17],radial_a_fe[17],  c=z, edgecolors='none')
	ax1.set_ylabel(r'$[Mg/Fe]$')
	ax7.set_ylabel(r'$[Mg/Fe]$')
	ax13.set_ylabel(r'$[Mg/Fe]$')
	ax13.set_xlabel(r'$[Fe/H]$')
	ax14.set_xlabel(r'$[Fe/H]$')
	ax15.set_xlabel(r'$[Fe/H]$')
	ax16.set_xlabel(r'$[Fe/H]$')
	ax17.set_xlabel(r'$[Fe/H]$')
	ax18.set_xlabel(r'$[Fe/H]$')
	axes = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18]
	for ax in axes:
		ax.set_xlim([-2.3,1.3])
		ax.set_ylim([-1.0,1.0])
	f.subplots_adjust(hspace=0.40, wspace=0, left=0.08, right=0.97, top=0.95)
	f.set_size_inches(12,6, forward=True)
	ax13.legend()

	ensure_dir(path+run+"_FOF"+str(halo)+"_SUB"+str(subgroup)+"/plots/")
	plttitle = path+run+"_FOF"+str(halo)+"_SUB"+str(subgroup)+"/plots/part"+str(parttype)+"alphafe.eps"

	plt.savefig(plttitle, format='eps', dpi=1200)

def plotmdf(datfile,halo,subgroup,parttype, path="/data5/astjmack/halos/"):
	array = np.load(datfile)
	R_pos = array[7]
	z_pos = array[8]
	fe_h = array[9]
	a_fe = array[10]
	r_200 = array[11]
	rad_bins = np.linspace(0.03*r_200,0.15*r_200,num=7, endpoint=False)
	z_bins = np.linspace(0.0,0.03*r_200,num=4, endpoint=False)
	
	radial_fe_h = []
	for j in range(0,len(z_bins)-1): 
		z_mask = [(np.abs(z_pos) > z_bins[j]) & (np.abs(z_pos) < z_bins[j+1])]
		fe_h_zbin = fe_h[z_mask]
		r_zbin = R_pos[z_mask]
		for i in range(0,len(rad_bins)-1):
			bin_mask = [(r_zbin > rad_bins[i]) & (r_zbin < rad_bins[i+1])]
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

	labels = [r'$0.03 < R < 0.05$ $R_{200}$', r'$0.05 < R < 0.07$ $R_{200}$', r'$0.07 < R < 0.09$ $R_{200}$', r'$0.09 < R < 0.11$ $R_{200}$', r'$0.11 < R < 0.13$ $R_{200}$', r'$0.13 < R < 0.15$ $R_{200}$']

	params = {'axes.labelsize': 14, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'text.usetex': True, 'lines.linewidth' : 2}
	plt.rcParams.update(params)
	fig, ax = plt.subplots(3,1, sharex=True)
	
	cmap = colors.LinearSegmentedColormap.from_list('sSFR',['red','green'],256)
	
	for i in range(0,(len(radial_fe_h)/3)):
		ax[2].plot(fe_h_centers[i], fe_h_hists[i], label=labels[i])
		ax[2].set_xlabel(r'$[Fe/H]$')
		ax[2].set_ylabel(r'$N/N*$')
		ax[2].set_title(r'$0.00 < |z| < 0.01 R_{200}$')
		ax[2].legend(loc=2, fontsize='x-small', frameon=False, shadow=False)
	for i in range((len(radial_fe_h)/3),(2*len(radial_fe_h)/3)):
		ax[1].plot(fe_h_centers[i], fe_h_hists[i], label=labels[i-6])
		ax[1].set_ylabel(r'$N/N*$')
		ax[1].set_title(r'$0.01 < |z| < 0.02 R_{200}$')
		ax[1].legend(loc=2, fontsize='x-small', frameon=False, shadow=False)
	for i in range((2*len(radial_fe_h)/3),(len(radial_fe_h))):
		ax[0].plot(fe_h_centers[i], fe_h_hists[i], label=labels[i-12])
		ax[0].set_ylabel(r'$N/N*$')
		ax[0].set_title(r'$0.02 < |z| < 0.03 R_{200}$')
		ax[0].legend(loc=2, fontsize='x-small', frameon=False, shadow=False)
	ensure_dir(path+run+"_FOF"+str(halo)+"_SUB"+str(subgroup)+"/plots/")
	plttitle = path+run+"_FOF"+str(halo)+"_SUB"+str(subgroup)+"/plots/part"+str(parttype)+"mdf.eps"
	plt.savefig(plttitle, format='eps', dpi=1200)

def selectdiskies(path="/data5/astjmack/halos/"):
	mass = E.readArray("SUBFIND", sim, tag, "/Subhalo/Stars/Mass") * 1e10
	group = E.readArray("SUBFIND", sim, tag, "/Subhalo/GroupNumber")
	subgroup = E.readArray("SUBFIND", sim, tag, "/Subhalo/SubGroupNumber")
	mask = [(mass > 1e10) & (mass < 3e10)]
	mass = mass[mask]
	group = group[mask]
	subgroup = subgroup[mask]
	array = np.array([group, subgroup, mass])
	print str(array)
