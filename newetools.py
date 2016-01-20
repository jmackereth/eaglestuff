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
import sys

default_run = "L0025N0376"
default_dir = "/data5/simulations/EAGLE/"
default_model = "REFERENCE"
default_tag = "028_z000p000"

verbose_option = False #Sets verbose = True/False for all reading

def select_sim(run=default_run,model=default_model,tag=default_tag,top_directory=default_dir):
	directory = top_directory + run + "/" + model + "/data"

	h = E.readAttribute("SUBFIND", directory, tag, "/Header/HubbleParam")
	masstable = E.readAttribute("SUBFIND", directory, tag, "/Header/MassTable") / h
	boxsize = E.readAttribute("SUBFIND", directory, tag, "/Header/BoxSize")
	boxsize = boxsize/h

	sim_info = [run, model, tag, top_directory, directory, h, masstable, boxsize]
	print "Selected %s - %s - %s" %(run,model,tag)
	return sim_info

def ensure_dir(f):
	""" Ensure a a file exists and if not make the relevant path """
	d = os.path.dirname(f)
	if not os.path.exists(d):
			os.makedirs(d)

abundance_path = "/PartType4/SmoothedElementAbundance/"

def loadparticles(simulation_info):
	""" This loads the particle data for a given simulation and returns an array with that data. """
	run, model, tag, top_directory, directory, h, masstable, boxsize = simulation_info
	print "Loading particle data for %s - %s - %s" %(run,model,tag)
	sim = directory
	print "Loading group numbers..." 
	groupnum_type = np.array( [E.readArray("PARTDATA", sim, tag, "/PartType0/GroupNumber",verbose=verbose_option), 
		E.readArray("PARTDATA", sim, tag, "/PartType1/GroupNumber",verbose=verbose_option), 
		E.readArray("PARTDATA", sim, tag, "/PartType4/GroupNumber",verbose=verbose_option), 
		E.readArray("PARTDATA", sim, tag, "/PartType5/GroupNumber",verbose=verbose_option)] )
	print "Loading subgroup numbers..."
	subgroupnum_type = np.array( [E.readArray("PARTDATA", sim, tag, "/PartType0/SubGroupNumber",verbose=verbose_option), 
		E.readArray("PARTDATA", sim, tag, "/PartType1/SubGroupNumber",verbose=verbose_option), 
		E.readArray("PARTDATA", sim, tag, "/PartType4/SubGroupNumber",verbose=verbose_option), 
		E.readArray("PARTDATA", sim, tag, "/PartType5/SubGroupNumber",verbose=verbose_option)] )
	print "Loading particle coordinates..."
	pos_type = np.array( [E.readArray("PARTDATA", sim, tag, "/PartType0/Coordinates",verbose=verbose_option), 
		E.readArray("PARTDATA", sim, tag, "/PartType1/Coordinates",verbose=verbose_option), 
		E.readArray("PARTDATA", sim, tag, "/PartType4/Coordinates",verbose=verbose_option), 
		E.readArray("PARTDATA", sim, tag, "/PartType5/Coordinates",verbose=verbose_option)] )
	print "Loading particle masses..."
	mass_type = np.array( [E.readArray("PARTDATA", sim, tag, "/PartType0/Mass",verbose=verbose_option), 
		(np.ones(len(pos_type[1]))*masstable[1]) , 
		E.readArray("PARTDATA", sim, tag, "/PartType4/Mass",verbose=verbose_option), 
		E.readArray("PARTDATA", sim, tag, "/PartType5/Mass",verbose=verbose_option)])
	print "Loading particle velocities..."
	vel_type = np.array( [E.readArray("PARTDATA", sim, tag, "/PartType0/Velocity",verbose=verbose_option), 
		E.readArray("PARTDATA", sim, tag, "/PartType1/Velocity",verbose=verbose_option), 
		E.readArray("PARTDATA", sim, tag, "/PartType4/Velocity",verbose=verbose_option), 
		E.readArray("PARTDATA", sim, tag, "/PartType5/Velocity",verbose=verbose_option)] )
	print "Loading particle abundances..."
	stars_abundances = np.array( [E.readArray("PARTDATA", sim, tag, abundance_path+"Hydrogen",verbose=verbose_option),  
		E.readArray("PARTDATA", sim, tag, abundance_path+"Helium",verbose=verbose_option), 
		E.readArray("PARTDATA", sim, tag, abundance_path+"Carbon",verbose=verbose_option), 
		E.readArray("PARTDATA", sim, tag, abundance_path+"Nitrogen",verbose=verbose_option), 
		E.readArray("PARTDATA", sim, tag, abundance_path+"Oxygen",verbose=verbose_option), 
		E.readArray("PARTDATA", sim, tag, abundance_path+"Neon",verbose=verbose_option), 
		E.readArray("PARTDATA", sim, tag, abundance_path+"Magnesium",verbose=verbose_option), 
		E.readArray("PARTDATA", sim, tag, abundance_path+"Silicon",verbose=verbose_option), 
		E.readArray("PARTDATA", sim, tag, abundance_path+"Iron",verbose=verbose_option)])
	print "Done loading."
	return np.array([groupnum_type, subgroupnum_type, pos_type, mass_type, vel_type, stars_abundances])
	

def loadfofdat(simulation_info):
	""" Load Relevant FOF data """
	run, model, tag, top_directory, directory, h, masstable, boxsize = simulation_info
	sim = directory
	print "Loading FoF data for %s - %s - %s" %(run,model,tag)
	fsid = np.array(E.readArray("SUBFIND_GROUP", sim, tag, "FOF/FirstSubhaloID",verbose=verbose_option))
	groupnumber = np.array(E.readArray("SUBFIND" , sim, tag, "/Subhalo/GroupNumber",verbose=verbose_option))[fsid]
	CoP = np.array(E.readArray("SUBFIND", sim, tag, "/Subhalo/CentreOfPotential",verbose=verbose_option))[fsid]
	subhalovel = np.array(E.readArray("SUBFIND", sim, tag, "/Subhalo/Velocity",verbose=verbose_option))[fsid]
	r_200 = np.array(E.readArray("SUBFIND_GROUP", sim, tag, "/FOF/Group_R_Crit200",verbose=verbose_option))
	tot_ang_mom = np.array(E.readArray("SUBFIND", sim, tag, "/Subhalo/Stars/Spin",verbose=verbose_option))[fsid]
	stellar_mass = np.array(E.readArray("SUBFIND", sim, tag, "/Subhalo/Stars/Mass",verbose=verbose_option) * 1e10)[fsid]
	stellar_abundances = np.array( [ E.readArray("SUBFIND", sim, tag, "/Subhalo/Stars/SmoothedElementAbundance/Hydrogen",verbose=verbose_option)[fsid], 
		E.readArray("SUBFIND", sim, tag, "/Subhalo/Stars/SmoothedElementAbundance/Helium",verbose=verbose_option)[fsid], 
		E.readArray("SUBFIND", sim, tag, "/Subhalo/Stars/SmoothedElementAbundance/Carbon",verbose=verbose_option)[fsid], 
		E.readArray("SUBFIND", sim, tag, "/Subhalo/Stars/SmoothedElementAbundance/Nitrogen",verbose=verbose_option)[fsid], 
		E.readArray("SUBFIND", sim, tag, "/Subhalo/Stars/SmoothedElementAbundance/Oxygen",verbose=verbose_option)[fsid], 
		E.readArray("SUBFIND", sim, tag, "/Subhalo/Stars/SmoothedElementAbundance/Neon",verbose=verbose_option)[fsid], 
		E.readArray("SUBFIND", sim, tag, "/Subhalo/Stars/SmoothedElementAbundance/Magnesium",verbose=verbose_option)[fsid], 
		E.readArray("SUBFIND", sim, tag, "/Subhalo/Stars/SmoothedElementAbundance/Silicon",verbose=verbose_option)[fsid], 
		E.readArray("SUBFIND", sim, tag, "/Subhalo/Stars/SmoothedElementAbundance/Iron",verbose=verbose_option)[fsid]])
	print "Done loading."
	return np.array([fsid,groupnumber,CoP,subhalovel,r_200,tot_ang_mom, stellar_mass, stellar_abundances ])

def stackparticles(partdat):
	""" Stack Particles ready for use in halo selection etc """
	print "Stacking particles..."
	parttype0 = np.zeros(len(partdat[0][0]))
	parttype1 = np.ones(len(partdat[0][1]))
	parttype4 = np.ones(len(partdat[0][2]))*4
	parttype5 = np.ones(len(partdat[0][3]))*5
	types = np.hstack((parttype0,parttype1,parttype4,parttype5))
	groupnums = np.hstack((partdat[0][0], partdat[0][1], partdat[0][2], partdat[0][3]))
	subgroupnums = np.hstack((partdat[1][0], partdat[1][1], partdat[1][2], partdat[1][3]))
	positions = np.hstack((partdat[2][0].T, partdat[2][1].T, partdat[2][2].T, partdat[2][3].T))
	starabundances = partdat[5]
	#print str(np.shape(starabundances))
	parttype0abunds = np.zeros((9,len(partdat[0][0])))
	parttype1abunds = np.zeros((9,len(partdat[0][1])))
	parttype5abunds = np.zeros((9,len(partdat[0][3])))
	abunds = np.hstack((parttype0abunds, parttype1abunds, starabundances, parttype5abunds))
	#positions = positions.T
	#print str(np.shape(positions))
	velocities = np.hstack((partdat[4][0].T, partdat[4][1].T, partdat[4][2].T, partdat[4][3].T))
	#velocities = velocities.T
	masses = np.hstack((partdat[3][0], partdat[3][1], partdat[3][2], partdat[3][3]))
	#print str(np.shape(masses))
	partarray = np.dstack((types,groupnums,subgroupnums,positions[0],positions[1], positions[2],velocities[0],velocities[1], velocities[2],masses, abunds[0], abunds[2], abunds[3], abunds[4], abunds[8]))
	partarray = partarray[0]
	print "Done stacking particles."
	return partarray

def loadsim(halonum=72,halo_function=False):
	siminfo = select_sim()
	partdat = loadparticles(siminfo)
	fofdat = loadfofdat(siminfo)
	partstack = stackparticles(partdat)
	if halo_function == True:
		halo(partstack,fofdat,halonum,siminfo)

def halo(partstack,fofdat,groupnum, simulation_info, plot=True, partdat_out=False, fofdat_out=False):
	""" define a central halo using groupnum and see its jz/jc histogram and morphology """
	print "Aligning..."
	boxsize = simulation_info[7]
	stack = partstack[(partstack[:,1] == groupnum) & (partstack[:,2] == 0)]
	fofindex = np.where(fofdat[1] == groupnum)
	CoP = fofdat[2][fofindex]
	r200 = fofdat[4][fofindex]
	pos = stack[:,3:6]-(CoP-(boxsize/2))
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
	starmass = stack[:,9][stack[:,0] == 4]
	starpos = pos[stack[:,0] == 4]
	starradii = np.linalg.norm(starpos, axis = 1)
	radmask = starradii < 0.15*r200
	starvel = stack[:,6:9][stack[:,0] == 4]-subhalovel
	massvel = np.array([starvel[i]*starmass[i] for i in range(0,len(starvel))])
	#print starpos,massvel
	starj = np.array([np.cross(starp,starv) for starp,starv in zip(starpos,massvel)])
	r200j = starj[radmask]
	tot_ang_mom = np.sum(r200j, axis = 0)
	a = np.matrix([tot_ang_mom[0],tot_ang_mom[1],tot_ang_mom[2]])/np.linalg.norm([tot_ang_mom[0],tot_ang_mom[1],tot_ang_mom[2]])
	b = np.matrix([0,0,1])
	v = np.cross(a,b)
	s = np.linalg.norm(v)
	c = np.dot(a,b.T)
	vx = np.matrix([[0,-v[0,2],v[0,1]],[v[0,2],0,-v[0,0]],[-v[0,1],v[0,0],0]])
	transform = np.eye(3,3) + vx + (vx*vx)*((1-c[0,0])/s**2)
	starpos = np.array([np.matmul(transform,starpos[i].T) for i in range(0,len(starpos))])[:,0]
	starvel = np.array([np.matmul(transform,starvel[i].T) for i in range(0,len(starvel))])[:,0]
	starr_xy = np.linalg.norm(np.dstack((starpos[:,0],starpos[:,1]))[0], axis = 1)
	G = 4.302e2
	starv_c = np.sqrt((G*starinnermass)/starr_xy)
	massvel = np.array([starvel[i]*starmass[i] for i in range(0,len(starvel))])
	starj = np.array([np.cross(starp,starv) for starp,starv in zip(starpos,massvel)])
	starjspec = np.array([np.cross(starp,starv) for starp,starv in zip(starpos,starvel)])
	starradii = np.linalg.norm(starpos, axis = 1)
	radmask = starradii < 0.15*r200 
	r200j = starj[radmask]
	tot_ang_mom = np.sum([r200j[:,0],r200j[:,1],r200j[:,2]], axis =1)
	tot_ang_mom = tot_ang_mom/np.linalg.norm(tot_ang_mom)
	#print str(tot_ang_mom)
	starj_z = starjspec[:,2]
	starj_c = starv_c*starr_xy
	starjz_jc = (starj_z/starj_c)
	
	
	starmass = stack[:,9][stack[:,0] == 4]
	if plot == True:
		hist, bins = np.histogram(starjz_jc, bins=100, range=(-2,2))
		centers = (bins[:-1] + bins[1:]) / 2
		hist = [float(histo)/float(sum(hist)) for histo in hist]
		xbins = np.arange(0,0.05,0.001)
		ybins = np.arange(-2,2,0.05)
		H, xedges, yedges = np.histogram2d(starjz_jc,starr_xy,bins=(ybins,xbins))
		my_cmap = matplotlib.cm.get_cmap('jet')
		my_cmap.set_under('w')
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
		
	
	if plot == True:
		x = starpos[:,0]
		y = starpos[:,1]
		z = starpos[:,2]
		params = {'axes.labelsize': 14, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'text.usetex': True, 'lines.linewidth' : 2}
		plt.rcParams.update(params)
		from mpl_toolkits.mplot3d import Axes3D
		fig = plt.figure()
		my_cmap = matplotlib.cm.get_cmap('jet')
		my_cmap.set_under('w')
		ybins = np.arange(-0.06, 0.06, 0.001)
		xbins = np.arange(-0.06, 0.06, 0.001)
		my_clim = (0,150)
		gs = gridspec.GridSpec(2,2)
		ax1 = fig.add_subplot(gs[0,0], projection='3d')
		ax1.scatter(x,y,z, s=starmass+5)
		ax2 = fig.add_subplot(gs[0,1])
		H, xedges, yedges = np.histogram2d(y,x,bins=(ybins,xbins))
		ax2.imshow(H, extent=[-0.06,0.06,-0.06,0.06], interpolation='nearest', origin='lower', aspect='auto',cmap=my_cmap, vmin=0.1, clim=my_clim)
		#ax2.scatter(x,y, s=starmass+10)
		ax2.set_xlabel(r'$x[Mpc]$')
		ax2.set_ylabel(r'$y[Mpc]$')
		ax3 = fig.add_subplot(gs[1,0])
		H, xedges, yedges = np.histogram2d(z,x,bins=(ybins,xbins))
		ax3.imshow(H, extent=[-0.06,0.06,-0.06,0.06], interpolation='nearest', origin='lower', aspect='auto',cmap=my_cmap, vmin=0.1, clim=my_clim)
		#ax3.scatter(x,z, s=starmass+10)
		ax3.set_xlabel(r'$x[Mpc]$')
		ax3.set_ylabel(r'$z[Mpc]$')
		ax4 = fig.add_subplot(gs[1,1])
		H, xedges, yedges = np.histogram2d(z,y,bins=(ybins,xbins))
		ax4.imshow(H, extent=[-0.06,0.06,-0.06,0.06], interpolation='nearest', origin='lower', aspect='auto',cmap=my_cmap, vmin=0.1, clim=my_clim)
		#ax4.scatter(y,z, s=starmass+10)
		ax4.set_xlabel(r'$y[Mpc]$')
		ax4.set_ylabel(r'$z[Mpc]$')
		fig.subplots_adjust(hspace=0.40, wspace=0.40, right=0.95, top=0.95)
		fof = "FOF "+ str(groupnum)
		fig.text(0.5, 0.96, fof)
		
	
	stars_h = stack[:,10][stack[:,0] == 4]
	stars_fe = stack[:,14][stack[:,0] == 4]
	stars_o = stack[:,13][stack[:,0] == 4]
	solar_h = 0.706498
	solar_fe = 0.00110322
	solar_o = 0.00549262
	solar_fe_h = np.log10(solar_fe/solar_h)
	solar_o_fe = np.log10(solar_o/solar_h)-(solar_fe_h)
	stars_fe_h = np.log10(stars_fe/stars_h)
	stars_o_fe = np.log10(stars_o/stars_h)-(stars_fe_h)
	fe_h = np.array([str_fe_h - solar_fe_h for str_fe_h in stars_fe_h])
	o_fe = np.array([str_o_fe - solar_o_fe for str_o_fe in stars_o_fe])

	if plot == True:
		R_pos = starr_xy
		z_pos = starpos[:,2]
		a_fe = o_fe
		r_200 = r200
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
		my_cmap = matplotlib.cm.get_cmap('jet')
		my_cmap.set_under('w')
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
		labels = [r'$'+str(rad_bins[0]*1000)+' < R < '+str(rad_bins[1]*1000)+'$ $Kpc$', r'$'+str(rad_bins[1]*1000)+' < R < '+str(rad_bins[2]*1000)+'$ $Kpc$', r'$'+str(rad_bins[2]*1000)+' < R < '+str(rad_bins[3]*1000)+'$ $Kpc$',	r'$'+str(rad_bins[3]*1000)+' < R < '+str(rad_bins[4]*1000)+'$ $Kpc$', r'$'+str(rad_bins[4]*1000)+' < R < '+str(rad_bins[5]*1000)+'$ $Kpc$', r'$'+str(rad_bins[5]*1000)+' < R < '+str(rad_bins[6]*1000)+'$ $Kpc$']
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
		

	if plot == True:
		

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
	
		labels = [r'$'+str(rad_bins[0]*1000)+' < R < '+str(rad_bins[1]*1000)+'$ $Kpc$', r'$'+str(rad_bins[1]*1000)+' < R < '+str(rad_bins[2]*1000)+'$ $Kpc$', r'$'+str(rad_bins[2]*1000)+' < R < '+str(rad_bins[3]*1000)+'$ 	$Kpc$', r'$'+str(rad_bins[3]*1000)+' < R < '+str(rad_bins[4]*1000)+'$ $Kpc$', r'$'+str(rad_bins[4]*1000)+' < R < '+str(rad_bins[5]*1000)+'$ $Kpc$', r'$'+str(rad_bins[5]*1000)+' < R < '+str(rad_bins[6]*1000)+'$ $Kpc$']
	
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
		
		plt.show()

	if partdat_out == True and fofdat_out == False:
		partarray = np.dstack((stack[:,0][stack[:,0] == 4], starpos[:,0], starpos[:,1], starpos[:,2], starvel[:,0], starvel[:,1], starvel[:,2], starmass, fe_h, o_fe, starj_z, starj_c, starjz_jc))[0]
		return partarray
	if fofdat_out == True and partdat_out == False:
		jz_jcdisky = float(len(starjz_jc[(starjz_jc < 1.2) & (starjz_jc > 0.7)]))
		lenjz_jc = float(len(starjz_jc))
		jz_jcdiskratio = jz_jcdisky/lenjz_jc
		
		n_highofe = float(len(o_fe[o_fe > 0.2]))
		n_lowofe = float(len(o_fe[o_fe < 0.2]))
		low_high_o_fe = n_highofe/n_lowofe
		high_total_o_fe = n_highofe/lenjz_jc
		fof_h = fofdat[7][0][fofindex]
		fof_fe = fofdat[7][8][fofindex]
		fof_fe_h = np.log10(fof_fe/fof_h)-solar_fe_h
		fof_stellar_mass = fofdat[6][fofindex]
		fofarray = np.array([groupnum, fof_stellar_mass, fof_fe_h, low_high_o_fe, high_total_o_fe, jz_jcdiskratio])
		return fofarray
	if partdat_out == True and fofdat_out == True:
		partarray = np.dstack((stack[:,0][stack[:,0] == 4], starpos[:,0], starpos[:,1], starpos[:,2], starvel[:,0], starvel[:,1], starvel[:,2], starmass, fe_h, o_fe, starj_z, starj_c, starjz_jc))[0]
		jz_jcdisky = float(len(((starjz_jc < 1.2) & (starjz_jc > 0.7))))
		lenjz_jc = float(len(starjz_jc))
		
		jz_jcdiskratio = jz_jcdisky/lenjz_jc
		n_highofe = float(len(o_fe > 0.2))
		n_lowofe = float(len(o_fe < 0.2))
		low_high_o_fe = n_highofe/n_lowofe
		high_total_o_fe = n_highofe/lenjz_jc
		fof_h = fofdat[7][0][fofindex]
		fof_fe = fofdat[7][8][fofindex]
		fof_fe_h = np.log10(fof_fe/fof_h)-solar_fe_h
		fof_stellar_mass = fofdat[6][fofindex]
		fofarray = np.array(groupnum, fof_stellar_mass, fof_fe_h, low_high_o_fe, high_total_o_fe, jz_jcdiskratio)
		return partarray, fofarray
		
def metallicity_gradient(partstack):
	print "To do"
	#To do