import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import colors
from matplotlib import gridspec
from scipy.stats import mode
from scipy.optimize import curve_fit
from scipy.integrate import quad
import numpy as np
import eagle as E
import os
import math
import csv
import sys
import h5py
import cPickle as pickle
import glob

default_run = "L0050N0752"
default_dir = "/data5/simulations/EAGLE/"
default_model = "REFERENCE"
default_tag = "028_z000p000"
default_sim = default_dir + default_run + "/" + default_model + "/data"

work_dir = '/data5/astjs/'
plot_dir = work_dir+'fofplots/'

def ensure_dir(f):
	""" Ensure a a file exists and if not make the relevant path """
	d = os.path.dirname(f)
	if not os.path.exists(d):
			os.makedirs(d)

def loadparticles(run=default_run,tag=default_tag,model=default_model,directory=default_dir, mass_cut=[7e11,3e12]):
	""" 
	This loads the particle and FoF data (and simulation attributes) for a given simulation , and given halo stellar mass range and returns arrays with that data.

	INPUT 
	run - the run which is to be extracted e.g. L0050N0752 for the 50Mpc run
	tag - the tag to extract e.g. 028_z000p000
	model - the model extracted e.g. REFERENCE
	directory - the directory which contains the necessary snapshot
	mass_cut - a tuple which gives the upper and lower limit for the mass range to be extracted (in Msun)
	
	OUTPUT
	partarray - an array of shape (18,N_particles) which contains necessary info for analysing haloes
	fofarray - an array of shape (7, N_FoFs) which gives information on all the FoFs in the simulation (NO MASS CUT APPLIED)
	simattributes - an array of shape (5) which gives information on the selected simulation and mass cut implied on partarray
	 """
	print "Loading particle data for %s - %s - %s" %(run,model,tag)
	sim = default_dir + run + "/" + model + "/data"
	print "Loading Simulation Attributes.."
	
	boxsize = E.readAttribute("SUBFIND", sim, tag, "/Header/BoxSize")
	h = E.readAttribute("SUBFIND", sim, tag, "/Header/HubbleParam")
	Omega0 = E.readAttribute("PARTDATA", sim, tag, "/Header/Omega0")
	OmegaLambda = E.readAttribute("PARTDATA", sim, tag, "/Header/OmegaLambda")
	OmegaBaryon = E.readAttribute("PARTDATA", sim, tag, "/Header/OmegaBaryon")
	a_0 = E.readAttribute("PARTDATA", sim, tag, "/Header/ExpansionFactor")
	boxsize = boxsize/h
	masstable = E.readAttribute("PARTDATA", sim, tag, "/Header/MassTable") / h
	simattributes = [run, model,  tag, h, boxsize, masstable, mass_cut, Omega0, OmegaLambda, OmegaBaryon, a_0]

	print "Loading FoF Data... "
	fsid = np.array(E.readArray("SUBFIND_GROUP", sim, tag, "FOF/FirstSubhaloID"))
	groupnumber = np.array(E.readArray("SUBFIND" , sim, tag, "/Subhalo/GroupNumber"))[fsid]
	CoP = np.array(E.readArray("SUBFIND", sim, tag, "/Subhalo/CentreOfPotential"))[fsid]
	subhalovel = np.array(E.readArray("SUBFIND", sim, tag, "/Subhalo/Velocity"))[fsid]
	r_200 = np.array(E.readArray("SUBFIND_GROUP", sim, tag, "/FOF/Group_R_Crit200"))
	m_200 = np.array(E.readArray("SUBFIND_GROUP", sim, tag, "/FOF/Group_M_Crit200")*1e10)
	tot_ang_mom = np.array(E.readArray("SUBFIND", sim, tag, "/Subhalo/Stars/Spin"))[fsid]
	stellar_mass = np.array(E.readArray("SUBFIND", sim, tag, "/Subhalo/Stars/Mass") * 1e10)[fsid]
	stellar_abundances = np.array( [ E.readArray("SUBFIND", sim, tag, "/Subhalo/Stars/SmoothedElementAbundance/Hydrogen")[fsid], E.readArray("SUBFIND", sim, tag, "/Subhalo/Stars/SmoothedElementAbundance/Helium")[fsid], E.readArray("SUBFIND", sim, tag, "/Subhalo/Stars/SmoothedElementAbundance/Carbon")[fsid], E.readArray("SUBFIND", sim, tag, "/Subhalo/Stars/SmoothedElementAbundance/Nitrogen")[fsid], E.readArray("SUBFIND", sim, tag, "/Subhalo/Stars/SmoothedElementAbundance/Oxygen")[fsid], E.readArray("SUBFIND", sim, tag, "/Subhalo/Stars/SmoothedElementAbundance/Neon")[fsid], E.readArray("SUBFIND", sim, tag, "/Subhalo/Stars/SmoothedElementAbundance/Magnesium")[fsid], E.readArray("SUBFIND", sim, tag, "/Subhalo/Stars/SmoothedElementAbundance/Silicon")[fsid], E.readArray("SUBFIND", sim, tag, "/Subhalo/Stars/SmoothedElementAbundance/Iron")[fsid]]) 
	fofarray = np.array([fsid,groupnumber,CoP,subhalovel,r_200,tot_ang_mom, stellar_mass, stellar_abundances, m_200 ])	

	locs = np.where(((fofarray[8] > mass_cut[0]) & (fofarray[8] < mass_cut[1])))
	mass_cut_groupnums = fofarray[1][locs]
	print "Loading subgroup numbers..."
	subgroupnum_type = np.array( [E.readArray("PARTDATA", sim, tag, "/PartType0/SubGroupNumber"), 
				      E.readArray("PARTDATA", sim, tag, "/PartType1/SubGroupNumber"), 
				      E.readArray("PARTDATA", sim, tag, "/PartType4/SubGroupNumber"), 
				      E.readArray("PARTDATA", sim, tag, "/PartType5/SubGroupNumber")] )
	print 'stripping satellites..'
	type0mask = subgroupnum_type[0] == 0
	type1mask = subgroupnum_type[1] == 0
	type4mask = subgroupnum_type[2] == 0
	type5mask = subgroupnum_type[3] == 0
	
	print str(len(subgroupnum_type[1]))+ ' ' +str(len(np.where(type1mask == True)[0]))
	print "Loading group numbers..." 
	groupnum_type = np.array( [E.readArray("PARTDATA", sim, tag, "/PartType0/GroupNumber"), 
				   E.readArray("PARTDATA", sim, tag, "/PartType1/GroupNumber"), 
				   E.readArray("PARTDATA", sim, tag, "/PartType4/GroupNumber"),
				   E.readArray("PARTDATA", sim, tag, "/PartType5/GroupNumber")] )
	
	type0groupmask = np.in1d(groupnum_type[0], mass_cut_groupnums)
	type1groupmask = np.in1d(groupnum_type[1], mass_cut_groupnums)
	type4groupmask = np.in1d(groupnum_type[2], mass_cut_groupnums)
	type5groupmask = np.in1d(groupnum_type[3], mass_cut_groupnums)

	type0mask = (type0mask & type0groupmask)
	type1mask = (type1mask & type1groupmask)
	type4mask = (type4mask & type4groupmask)
	type5mask = (type5mask & type5groupmask)

	subgroupnum_type = np.array([subgroupnum_type[0][type0mask], subgroupnum_type[1][type1mask], subgroupnum_type[2][type4mask], subgroupnum_type[3][type5mask]])
	groupnum_type = np.array([groupnum_type[0][type0mask], groupnum_type[1][type1mask], groupnum_type[2][type4mask], groupnum_type[3][type5mask]])
	print "Loading particle coordinates..."
	pos_type = np.array( [E.readArray("PARTDATA", sim, tag, "/PartType0/Coordinates")[type0mask], 
			      E.readArray("PARTDATA", sim, tag, "/PartType1/Coordinates")[type1mask],
			      E.readArray("PARTDATA", sim, tag, "/PartType4/Coordinates")[type4mask],
			      E.readArray("PARTDATA", sim, tag, "/PartType5/Coordinates")[type5mask]] )
	print "Loading particle masses..."
	mass_type = np.array( [(E.readArray("PARTDATA", sim, tag, "/PartType0/Mass")[type0mask]*1e10), 
			       (np.ones(len(pos_type[1]))*masstable[1]*1e10) , 
			       (E.readArray("PARTDATA", sim, tag, "/PartType4/Mass")[type4mask]*1e10),
			       (E.readArray("PARTDATA", sim, tag, "/PartType5/Mass")[type5mask]*1e10)])
	print "Loading particle velocities..."
	vel_type = np.array( [E.readArray("PARTDATA", sim, tag, "/PartType0/Velocity")[type0mask], 
			      E.readArray("PARTDATA", sim, tag, "/PartType1/Velocity")[type1mask],	
			      E.readArray("PARTDATA", sim, tag, "/PartType4/Velocity")[type4mask],
			      E.readArray("PARTDATA", sim, tag, "/PartType5/Velocity")[type5mask]] )
	print "Loading particle abundances..."
	stars_abundances = np.array( [E.readArray("PARTDATA", sim, tag, "/PartType4/SmoothedElementAbundance/Hydrogen")[type4mask], 
								  E.readArray("PARTDATA", sim, tag, "/PartType4/SmoothedElementAbundance/Helium")[type4mask], 
								  E.readArray("PARTDATA", sim, tag, "/PartType4/SmoothedElementAbundance/Carbon")[type4mask], 
								  E.readArray("PARTDATA", sim, tag, "/PartType4/SmoothedElementAbundance/Nitrogen")[type4mask], 
								  E.readArray("PARTDATA", sim, tag, "/PartType4/SmoothedElementAbundance/Oxygen")[type4mask], 
								  E.readArray("PARTDATA", sim, tag, "/PartType4/SmoothedElementAbundance/Neon")[type4mask], 
								  E.readArray("PARTDATA", sim, tag, "/PartType4/SmoothedElementAbundance/Magnesium")[type4mask], 
								  E.readArray("PARTDATA", sim, tag, "/PartType4/SmoothedElementAbundance/Silicon")[type4mask], 
								  E.readArray("PARTDATA", sim, tag, "/PartType4/SmoothedElementAbundance/Iron")[type4mask]])
	print "Loading Star formation expansion factors..."
	starformtime = np.array(E.readArray("PARTDATA", sim, tag, "/PartType4/StellarFormationTime")[type4mask])
	print "Loading Particle ID's..."
	PIDS = np.hstack((np.array(E.readArray("PARTDATA", sim, tag, "/PartType0/ParticleIDs"))[type0mask],
				np.array(E.readArray("PARTDATA", sim, tag, "/PartType1/ParticleIDs"))[type1mask],
				np.array(E.readArray("PARTDATA", sim, tag, "/PartType4/ParticleIDs"))[type4mask],
				np.array(E.readArray("PARTDATA", sim, tag, "/PartType5/ParticleIDs"))[type5mask]))
	print "Done loading."
	
	print "Making Particle Stack..."
	groupnumtypelens = [len(groupnum_type[0]), len(groupnum_type[1]), len(groupnum_type[2]), len(groupnum_type[3])] 
	parttype0 = np.zeros(groupnumtypelens[0])
	parttype1 = np.ones(groupnumtypelens[1])
	parttype4 = np.ones(groupnumtypelens[2])*4
	parttype5 = np.ones(groupnumtypelens[3])*5
	types = np.hstack((parttype0,parttype1,parttype4,parttype5))
	del parttype0, parttype1, parttype4, parttype5
	groupnums = np.hstack((groupnum_type[0], groupnum_type[1], groupnum_type[2], groupnum_type[3]))
	del groupnum_type
	subgroupnums = np.hstack((subgroupnum_type[0], subgroupnum_type[1], subgroupnum_type[2], subgroupnum_type[3]))
	del subgroupnum_type
	positions = np.hstack((pos_type[0].T, pos_type[1].T, pos_type[2].T, pos_type[3].T))
	del pos_type
	parttype0abunds = np.zeros((9,groupnumtypelens[0]))
	parttype1abunds = np.zeros((9,groupnumtypelens[1]))
	parttype5abunds = np.zeros((9,groupnumtypelens[3]))
	abunds = np.hstack((parttype0abunds, parttype1abunds, stars_abundances, parttype5abunds))
	del parttype0abunds, parttype1abunds, parttype5abunds, stars_abundances
	velocities = np.hstack((vel_type[0].T, vel_type[1].T, vel_type[2].T, vel_type[3].T))
	del vel_type
	masses = np.hstack((mass_type[0], mass_type[1], mass_type[2], mass_type[3]))
	del mass_type
	formtimes = np.hstack((np.zeros(groupnumtypelens[0]), np.zeros(groupnumtypelens[1]), starformtime, np.zeros(groupnumtypelens[3])))
	del starformtime
	print 'STAAAAAACK!'
	#print '%s %s %s %s %s %s %s' %(types.shape, groupnums.shape, subgroupnums.shape,  positions.shape, velocities.shape, abunds.shape, masses.shape)
	partarray = np.dstack((types,groupnums,subgroupnums,positions[0],positions[1], positions[2],velocities[0],velocities[1], velocities[2], masses, abunds[0], abunds[1], abunds[2], abunds[3], abunds[4], abunds[5], abunds[6], abunds[7], abunds[8], formtimes, PIDS))
	partarray = partarray[0]
	
	return partarray, fofarray, simattributes

def loadsnipparticles( groupnum, run=default_run,tag=default_tag,model=default_model,directory=default_dir):
	print "Loading particle data for snipshot %s - %s - %s" %(run,model,tag)
	sim = default_dir + run + "/" + model + "/data"
	boxsize = E.readAttribute("SNIP_PARTDATA", sim, tag, "/Header/BoxSize")
	h = E.readAttribute("SNIP_PARTDATA", sim, tag, "/Header/HubbleParam")
	Omega0 = E.readAttribute("SNIP_PARTDATA", sim, tag, "/Header/Omega0")
	OmegaLambda = E.readAttribute("SNIP_PARTDATA", sim, tag, "/Header/OmegaLambda")
	OmegaBaryon = E.readAttribute("SNIP_PARTDATA", sim, tag, "/Header/OmegaBaryon")
	a_0 = E.readAttribute("SNIP_PARTDATA", sim, tag, "/Header/ExpansionFactor")
	boxsize = boxsize/h
	masstable = E.readAttribute("SNIP_PARTDATA", sim, tag, "/Header/MassTable") / h
	simattributes = [run, model,  tag, h, boxsize, masstable, 0, Omega0, OmegaLambda, OmegaBaryon, a_0]
	numparts = E.readAttribute("SNIP_PARTDATA", sim, tag, "/Header/NumPart_Total")
	print sum(numparts)
	if numparts[5] == 0:
		print 'no black holes, ye have gone too far...'
		return None

	
	fsid = np.array(E.readArray("SNIP_SUBFIND", sim, tag, "FOF/FirstSubhaloID"))[groupnum-1]
	groupnumber = np.array(E.readArray("SNIP_SUBFIND" , sim, tag, "/Subhalo/GroupNumber"))
	group = groupnumber == groupnum
	CoP = np.array(E.readArray("SNIP_SUBFIND", sim, tag, "/Subhalo/CentreOfPotential"))[group]
	subhalovel = np.array(E.readArray("SNIP_SUBFIND", sim, tag, "/Subhalo/Velocity"))[group]
	r_200 = np.array(E.readArray("SNIP_SUBFIND", sim, tag, "/FOF/Group_R_Crit200"))[groupnum-1]
	tot_ang_mom = np.array(E.readArray("SNIP_SUBFIND", sim, tag, "/Subhalo/Stars/Spin"))[group]
	stellar_mass = np.array(E.readArray("SNIP_SUBFIND", sim, tag, "/Subhalo/Stars/Mass") * 1e10)[group]
	stellar_abundances = np.array( [ E.readArray("SNIP_SUBFIND", sim, tag, "/Subhalo/Stars/SmoothedElementAbundance/Hydrogen")[group], E.readArray("SNIP_SUBFIND", sim, tag, "/Subhalo/Stars/SmoothedElementAbundance/Helium")[group], E.readArray("SNIP_SUBFIND", sim, tag, "/Subhalo/Stars/SmoothedElementAbundance/Carbon")[group], E.readArray("SNIP_SUBFIND", sim, tag, "/Subhalo/Stars/SmoothedElementAbundance/Nitrogen")[group], E.readArray("SNIP_SUBFIND", sim, tag, "/Subhalo/Stars/SmoothedElementAbundance/Oxygen")[group], E.readArray("SNIP_SUBFIND", sim, tag, "/Subhalo/Stars/SmoothedElementAbundance/Neon")[group], E.readArray("SNIP_SUBFIND", sim, tag, "/Subhalo/Stars/SmoothedElementAbundance/Magnesium")[group], E.readArray("SNIP_SUBFIND", sim, tag, "/Subhalo/Stars/SmoothedElementAbundance/Silicon")[group], E.readArray("SNIP_SUBFIND", sim, tag, "/Subhalo/Stars/SmoothedElementAbundance/Iron")[group]]) 
	fofarray = [fsid,groupnumber,CoP,subhalovel,r_200,tot_ang_mom, stellar_mass, stellar_abundances ]
	
	
	print "Loading subgroup numbers..."
	subgroupnum_type = np.array( [E.readArray("SNIP_PARTDATA", sim, tag, "/PartType0/SubGroupNumber"), 
				      E.readArray("SNIP_PARTDATA", sim, tag, "/PartType1/SubGroupNumber"), 
				      E.readArray("SNIP_PARTDATA", sim, tag, "/PartType4/SubGroupNumber"), 
				      E.readArray("SNIP_PARTDATA", sim, tag, "/PartType5/SubGroupNumber")] )
	print 'stripping satellites..'
	type0mask = subgroupnum_type[0] == 0
	type1mask = subgroupnum_type[1] == 0
	type4mask = subgroupnum_type[2] == 0
	type5mask = subgroupnum_type[3] == 0
	
	print str(len(subgroupnum_type[1]))+ ' ' +str(len(np.where(type1mask == True)[0]))
	print "Loading group numbers..." 
	groupnum_type = np.array( [E.readArray("SNIP_PARTDATA", sim, tag, "/PartType0/GroupNumber"), 
				   E.readArray("SNIP_PARTDATA", sim, tag, "/PartType1/GroupNumber"), 
				   E.readArray("SNIP_PARTDATA", sim, tag, "/PartType4/GroupNumber"),
				   E.readArray("SNIP_PARTDATA", sim, tag, "/PartType5/GroupNumber")] )
	
	type0groupmask = groupnum_type[0] == groupnum
	type1groupmask = groupnum_type[1] == groupnum
	type4groupmask = groupnum_type[2] == groupnum
	type5groupmask = groupnum_type[3] == groupnum

	type0mask = (type0mask & type0groupmask)
	type1mask = (type1mask & type1groupmask)
	type4mask = (type4mask & type4groupmask)
	type5mask = (type5mask & type5groupmask)

	subgroupnums = np.hstack((subgroupnum_type[0][type0mask], subgroupnum_type[1][type1mask], subgroupnum_type[2][type4mask], subgroupnum_type[3][type5mask]))
	groupnums = np.hstack((groupnum_type[0][type0mask], groupnum_type[1][type1mask], groupnum_type[2][type4mask], groupnum_type[3][type5mask]))
	print "Loading particle coordinates..."
	positions = np.hstack( (E.readArray("SNIP_PARTDATA", sim, tag, "/PartType0/Coordinates")[type0mask].T, 
			      E.readArray("SNIP_PARTDATA", sim, tag, "/PartType1/Coordinates")[type1mask].T,
			      E.readArray("SNIP_PARTDATA", sim, tag, "/PartType4/Coordinates")[type4mask].T,
			      E.readArray("SNIP_PARTDATA", sim, tag, "/PartType5/Coordinates")[type5mask].T) )
	print "Loading particle masses..."
	masses = np.hstack((E.readArray("SNIP_PARTDATA", sim, tag, "/PartType0/Mass")[type0mask]*1e10, 
			       (np.ones(len(type1mask))*masstable[1]*1e10)[type1mask] , 
			       E.readArray("SNIP_PARTDATA", sim, tag, "/PartType4/Mass")[type4mask]*1e10,
			       E.readArray("SNIP_PARTDATA", sim, tag, "/PartType5/Mass")[type5mask]*1e10))
	print "Loading particle velocities..."
	velocities = np.hstack((E.readArray("SNIP_PARTDATA", sim, tag, "/PartType0/Velocity")[type0mask].T, 
			      E.readArray("SNIP_PARTDATA", sim, tag, "/PartType1/Velocity")[type1mask].T,	
			      E.readArray("SNIP_PARTDATA", sim, tag, "/PartType4/Velocity")[type4mask].T,
			      E.readArray("SNIP_PARTDATA", sim, tag, "/PartType5/Velocity")[type5mask].T) )
	
	print "Loading Star formation expansion factors..."
	starformtime = np.array(E.readArray("SNIP_PARTDATA", sim, tag, "/PartType4/StellarFormationTime")[type4mask])
	print "Loading Particle ID's..."
	PIDS = np.hstack((np.array(E.readArray("SNIP_PARTDATA", sim, tag, "/PartType0/ParticleIDs"))[type0mask],
				np.array(E.readArray("SNIP_PARTDATA", sim, tag, "/PartType1/ParticleIDs"))[type1mask],
				np.array(E.readArray("SNIP_PARTDATA", sim, tag, "/PartType4/ParticleIDs"))[type4mask],
				np.array(E.readArray("SNIP_PARTDATA", sim, tag, "/PartType5/ParticleIDs"))[type5mask]))
	print "Done loading."
	groupnumtypelens = [len(np.ones(len(type0mask))[type0mask]), len(np.ones(len(type1mask))[type1mask]), len(np.ones(len(type4mask))[type4mask]), len(np.ones(len(type5mask))[type5mask])] 
	parttype0 = np.zeros(groupnumtypelens[0])
	parttype1 = np.ones(groupnumtypelens[1])
	parttype4 = np.ones(groupnumtypelens[2])*4
	parttype5 = np.ones(groupnumtypelens[3])*5
	types = np.hstack((parttype0,parttype1,parttype4,parttype5))
	formtimes = np.hstack((np.zeros(groupnumtypelens[0]), np.zeros(groupnumtypelens[1]), starformtime, np.zeros(groupnumtypelens[3])))
	partarray = np.dstack((types, groupnums, subgroupnums, positions[0], positions[1], positions[2], velocities[0], velocities[1], velocities[2], masses, formtimes, PIDS))[0]
	return partarray, fofarray, simattributes

def halo(partstack,fofdat,simattributes,groupnum, snip=False):
	""" define a central halo using groupnum and see its jz/jc histogram and morphology """
	print 'Isolating FoF and Aligning....'
	#Isolate the desired Group
	stack = partstack[(partstack[:,1] == groupnum) & (partstack[:,2] == 0)]
	fofindex = np.where(fofdat[1] == groupnum)
	CoP = fofdat[2][fofindex]
	r200 = fofdat[4][fofindex]
	boxsize = simattributes[4]
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
	G = 4.302e-9  #Units?
	starv_c = np.sqrt((G*starinnermass)/starr_xy)
	massvel = np.array([starvel[i]*starmass[i] for i in range(0,len(starvel))])
	starj = np.array([np.cross(starp,starv) for starp,starv in zip(starpos,massvel)])
	starjspec = np.array([np.cross(starp,starv) for starp,starv in zip(starpos,starvel)])
	starradii = np.linalg.norm(starpos, axis = 1)
	radmask = starradii < 0.15*r200 
	r200j = starj[radmask]
	tot_ang_mom = np.sum([r200j[:,0],r200j[:,1],r200j[:,2]], axis =1)
	tot_ang_mom = tot_ang_mom/np.linalg.norm(tot_ang_mom)
	print 'aligned!     angular momentum:'+str(tot_ang_mom)
	starj_z = starjspec[:,2]
	starj_c = starv_c*starr_xy
	starjz_jc = (starj_z/starj_c)
	kappa = (0.5*starmass*((starj_z/starr_xy)**2))/(0.5*starmass*(np.linalg.norm(starvel, axis=1)**2))
	star_binding_energy = (0.6 * G * starinnermass**2)/starradii #Should probably not assume spherical potential
	fof_median_binding_energy = np.median(star_binding_energy)

	if snip == False:
		starmass = stack[:,9][stack[:,0] == 4]
		stars_h = stack[:,10][stack[:,0] == 4]
		stars_he =stack[:,11][stack[:,0] == 4]
		stars_fe = stack[:,18][stack[:,0] == 4]
		stars_c = stack[:,12][stack[:,0] == 4]
		stars_n = stack[:,13][stack[:,0] == 4]
		stars_o = stack[:,14][stack[:,0] == 4]
		stars_ne = stack[:,15][stack[:,0] == 4]
		stars_mg = stack[:,16][stack[:,0] == 4]
		stars_si = stack[:,17][stack[:,0] == 4]
		solar_h = 0.706498
		solar_fe = 0.00110322
		solar_o = 0.00549262
		solar_si = 0.000682587
		solar_mg = 0.000590706
		solar_fe_h = np.log10(solar_fe/solar_h)
		solar_o_h = np.log10(solar_o/solar_h)
		solar_si_h = np.log10(solar_si/solar_h)
		solar_mg_h = np.log10(solar_mg/solar_h)
		solar_o_fe = np.log10(solar_o/solar_h)-(solar_fe_h)
		solar_mg_fe = np.log10(solar_mg/solar_h)-(solar_fe_h)
		solar_si_fe = np.log10(solar_si/solar_h)-(solar_fe_h)
		solar_a_fe = np.log10(((solar_o+solar_si+solar_mg)/3)/solar_h)-(solar_fe_h)
		stars_fe_h = np.log10(stars_fe/stars_h)
		stars_o_h = np.log10(stars_o/stars_h)
		stars_si_h = np.log10(stars_si/stars_h)
		stars_mg_h = np.log10(stars_mg/stars_h)
		stars_c_fe = np.log10(stars_c/stars_h)-(stars_fe_h)
		stars_n_fe = np.log10(stars_n/stars_h)-(stars_fe_h)
		stars_o_fe = np.log10(stars_o/stars_h)-(stars_fe_h)
		stars_ne_fe = np.log10(stars_ne/stars_h)-(stars_fe_h)
		stars_mg_fe = np.log10(stars_mg/stars_h)-(stars_fe_h)
		stars_si_fe = np.log10(stars_si/stars_h)-(stars_fe_h)
		stars_a_fe = (np.log10(stars_o/stars_h)+np.log10(stars_si/stars_h)+np.log10(stars_mg/stars_h))-2*(stars_fe_h)
		o_h = np.array([str_o_h - solar_o_h for str_o_h in stars_o_h])
		si_h = np.array([str_si_h - solar_si_h for str_si_h in stars_si_h])
		mg_h = np.array([str_mg_h - solar_mg_h for str_mg_h in stars_mg_h])
		fe_h = np.array([str_fe_h - solar_fe_h for str_fe_h in stars_fe_h])
		solar_omgsi_h = np.log10((solar_o+solar_si+solar_mg)/3*solar_h)
		stars_omgsi_h = np.log10((stars_o+stars_si+stars_mg)/3*stars_h)
		omgsi_h = np.array([str_omgsi_h - solar_omgsi_h for str_omgsi_h in stars_omgsi_h])
		a_fe = omgsi_h - fe_h
	#a_fe = np.array([str_a_fe - solar_a_fe for str_a_fe in stars_a_fe])
	
	jz_jcdisky = float(len(starjz_jc[(starjz_jc > 0.7)]))
	lenjz_jc = float(len(starjz_jc))
	jz_jcdiskratio = jz_jcdisky/lenjz_jc
	if snip == False:
		rad_low = 0.003
		rad_high = 0.015
		rad_bins = np.linspace(rad_low,rad_high,num=15, endpoint=True)
		radial_fe_h = []
		radial_mass = []
		bincenters = []
		for i in range(0,len(rad_bins)-1):
				bin_mask = [(np.abs(starr_xy) > rad_bins[i]) & (np.abs(starr_xy) < rad_bins[i+1])]
				bincenter = rad_bins[i]+((rad_bins[i+1]-rad_bins[i])/2)
				fe_h_bin = fe_h[bin_mask]
				mass_bin = starmass[bin_mask]
				a_fe_bin = a_fe[bin_mask]
				bincenters.append(bincenter)
				radial_fe_h.append(fe_h_bin)
				radial_mass.append(mass_bin)
		fe_h_av = []
		for i in range(0,len(radial_fe_h)):
				av_fe_h = sum(np.array(radial_fe_h[i]*radial_mass[i])[radial_fe_h[i] != -np.inf])/sum(radial_mass[i])
				fe_h_av.append(av_fe_h)	
	
		fit = np.polyfit(bincenters, fe_h_av, 1)
		fe_h_grad = fit[0]
		z = np.poly1d(fit)
		newx = np.arange(bincenters[0], bincenters[-1], 0.001)
		n_highafe = float(len(a_fe[a_fe > 0.2]))
		n_lowafe = float(len(a_fe[a_fe < 0.2]))
		low_high_a_fe = n_highafe/n_lowafe
		high_total_a_fe = n_highafe/lenjz_jc
		fof_h = fofdat[7][0][fofindex]
		fof_fe = fofdat[7][8][fofindex]
		fof_fe_h = np.log10(fof_fe/fof_h)-solar_fe_h
		fof_stellar_mass = fofdat[6][fofindex]
		hist, bins = np.histogram(np.abs(starpos[:,0]), bins=100, range=(0,0.06))
		centers = (bins[:-1] + bins[1:]) / 2
		hist = hist

		def sersic(r, I_0, r_e, n):
			return I_0 * np.exp(-1*(r/r_e)**(1/n))
		try:
			popt, pcov = curve_fit(sersic, centers, hist, p0=[hist[0],0.01,1])
		except RuntimeError:
			print 'WARNING: could not fit sersic profile... returning NaN'
			popt = [np.nan, np.nan, np.nan]

		r_e = popt[1]
		sersic_index = popt[2]
		fe_h_grad = fe_h_grad*r_e
		formtimes = np.array(stack[:,19][stack[:,0]==4])
		age = expansion2age(simattributes, formtimes)
		PIDS = stack[:,20][stack[:,0]==4]

		disc_mask = (starjz_jc > 0.7) & (starjz_jc < 5) & (starradii > 0.002) & (starradii < 0.02) & (starpos[:,2] < 0.05) #Check
		halo_mask = [not i for i in disc_mask] & (star_binding_energy > fof_median_binding_energy) #Find halo stars
		mask1 = [not i for i in disc_mask]

		fofarray = np.array([groupnum, fof_stellar_mass, fof_fe_h, low_high_a_fe, high_total_a_fe, jz_jcdiskratio, fe_h_grad, r200, r_e, sersic_index, fof_median_binding_energy])
		partarray = np.dstack((stack[:,0][stack[:,0] == 4], starpos[:,0], starpos[:,1], starpos[:,2], starvel[:,0], starvel[:,1], starvel[:,2], starmass, fe_h, a_fe, starj_z, starj_c, starjz_jc, stars_h, stars_he, stars_c, stars_n, stars_o, stars_ne, stars_mg, stars_si, stars_fe, age, PIDS, kappa, star_binding_energy, halo_mask, disc_mask))[0]

	if snip == True:
		PIDS = stack[:,10][stack[:,0]==4]
		def exponential(z, N_0, z_s, n):
			return N_0 * np.exp(-1*(z/z_s)**n)
		disk = (starr_xy > 0.003) & (starr_xy < 0.015)
		z_disk = starpos[:,2][disk]
		mass_disk = starmass[disk]
		hist, bins = np.histogram(np.abs(z_disk), bins=100, range=(0,0.003),weights=mass_disk)
    		centers = (bins[:-1] + bins[1:]) / 2
		try:
    			popt, pcov = curve_fit(exponential, centers, hist, p0=[hist[0],0.002,1])
		except RuntimeError:
			print 'WARNING: could not fit scale height for halo '+ str(self.groupnumber)+'... check disk alignment?'
			return None
		scaleheight = popt[1]*1e3
		gal_a = np.array(simattributes[10])
		Omega0 = simattributes[7]
		OmegaLambda = simattributes[8]
		OmegaBaryon = simattributes[9]
		def t_lookback(a):
			return a/(np.sqrt(Omega0*a+OmegaLambda*(a**4)))
		htime =  (1/(simattributes[3]*100))*(3.086e19/3.1536e16)*quad(t_lookback,0,1)[0]
		gal_time = (1/(simattributes[3]*100))*(3.086e19/3.1536e16)*quad(t_lookback,0,gal_a)[0]
		gal_age = htime-gal_time
		fofarray = np.array([groupnum, jz_jcdiskratio, scaleheight, gal_age ])
		partarray = np.dstack((stack[:,0][stack[:,0] == 4], starpos[:,0], starpos[:,1], starpos[:,2], starvel[:,0], starvel[:,1], starvel[:,2], starmass, np.zeros(len(starpos[:,0])), np.zeros(len(starpos[:,0])), starj_z, starj_c, starjz_jc, PIDS, kappa))[0]
	return partarray, fofarray

def expansion2age(simattributes, formexpfactors):
	Omega0 = simattributes[7]
	OmegaLambda = simattributes[8]
	OmegaBaryon = simattributes[9]
	a_0 = simattributes[10]
	def t_lookback(a):
		return a/(np.sqrt(Omega0*a+OmegaLambda*(a**4)))
	formtimes = formexpfactors
	htime =  (1/(simattributes[3]*100))*(3.086e19/3.1536e16)*quad(t_lookback,0,a_0)[0]
	t_em = np.array([quad(t_lookback, formtime, a_0) for formtime in formtimes])
	t_em = t_em[:,0]
	age = t_em*htime
	return age

def find_nearest(array,value):
	idx = (np.abs(array-value)).argmin()
	return idx, array[idx]

def findnearestsnip(age):
	snip_ages = np.load(work_dir+'code/snip_400_z20p00_z00p00_htime_13.831.npy')
	snip_tags = np.load(work_dir+'code/snip_tags_L0025N0752_RECALIBRATED.npy')
	snip, age = find_nearest(snip_ages, age)
	tag = snip_tags[snip]
	return tag

def searchsnipforhalo(haloPIDs, simattributes, snip_tag):
	run = simattributes[0]
	model = simattributes[1]
	sim = default_dir + run + "/" + model + "/data"
	print str(sim)
	print len(haloPIDs)
	snip_groupnums = np.array(E.readArray("SNIP_PARTDATA", sim, snip_tag, "/PartType4/GroupNumber"))
	snip_PIDs = np.array(E.readArray("SNIP_PARTDATA", sim, snip_tag, "/PartType4/ParticleIDs"))
	PID_ind = np.in1d(snip_PIDs, haloPIDs)
	matched_groupnum = mode(snip_groupnums[PID_ind])[0][0]
	return matched_groupnum
	

def savehaloarrays(partarray, fofarray, simattributes, directory=work_dir):
	run, model,  tag, h, boxsize, masstable, mass_cut, Omega0, OmegaLambda, OmegaBaryon, a_0 = simattributes
	groupnum, fof_stellar_mass, fof_fe_h, low_high_a_fe, high_total_a_fe, jz_jcdiskratio, fe_h_grad, r200, r_e, sersic_index, fof_median_binding_energy = fofarray
	stack = partarray
	subfolder = 'savedhalos/%s/%s/%s/' %(run, model, tag)
	filename = directory+subfolder+run+'_'+model+'_'+tag+'_FOF'+str(int(groupnum))+'.hdf5'
	ensure_dir(directory+subfolder)
	f = h5py.File(filename, 'w')
	attrib_grp = f.create_group('simattributes')
	fof_grp = f.create_group('fofdata')
	stars_grp = f.create_group('stardata')
	run_data = attrib_grp.create_dataset('run', data=run)
	mod_data = attrib_grp.create_dataset('model', data=model)
	tag_data = attrib_grp.create_dataset('tag', data=tag)
	h_data = attrib_grp.create_dataset('h', data=h)
	boxsize_data = attrib_grp.create_dataset('boxsize', data=boxsize)
	masstable = attrib_grp.create_dataset('masstable', data=masstable)
	mass_cut = attrib_grp.create_dataset('mass_cut', data=mass_cut)
	Om0 = attrib_grp.create_dataset('Omega0', data=Omega0)
	OLambda = attrib_grp.create_dataset('OmegaLambda', data=OmegaLambda)
	OBaryon = attrib_grp.create_dataset('OmegaBaryon', data=OmegaBaryon)
	a0 = attrib_grp.create_dataset('ExpansionFactor', data=a_0)

	fofgnum = fof_grp.create_dataset('groupnum', data=groupnum)
	fofmass = fof_grp.create_dataset('stellarmass', data=fof_stellar_mass)
	foffeh = fof_grp.create_dataset('stellar_fe_h', data=fof_fe_h)
	lowhighoferatio = fof_grp.create_dataset('low_high_a_fe_ratio', data=low_high_a_fe)
	highofepercent = fof_grp.create_dataset('percenthigh_a_fe', data=high_total_a_fe)
	jz_jcratio = fof_grp.create_dataset('ratio_jz_jc_sim1', data=jz_jcdiskratio)
	fe_h_grad = fof_grp.create_dataset('fe_h_gradient', data=fe_h_grad)
	r_200fof = fof_grp.create_dataset('r200', data=r200)
	scalelength = fof_grp.create_dataset('scalelength', data=r_e)
	sersic = fof_grp.create_dataset('sersicindex', data=sersic_index)
	median_binding_energy = fof_grp.create_dataset('median_binding_energy', data=fof_median_binding_energy)

	abundances = stack[:,13:22]
	startype = stars_grp.create_dataset('type', data=stack[:,0])
	starpos = stars_grp.create_dataset('position', data=stack[:,1:4])
	starvel = stars_grp.create_dataset('velocity', data=stack[:,4:7])
	mass = stars_grp.create_dataset('mass', data=stack[:,7])
	starfe_h = stars_grp.create_dataset('fe_h', data=stack[:,8])
	staro_fe = stars_grp.create_dataset('a_fe', data=stack[:,9])
	star_abunds = stars_grp.create_dataset('abundances', data=abundances)
	star_jz = stars_grp.create_dataset('jz', data=stack[:,10])
	star_jc = stars_grp.create_dataset('jc', data=stack[:,11])
	star_jz_jc = stars_grp.create_dataset('jz_jc', data=stack[:,12])
	star_age = stars_grp.create_dataset('age', data=stack[:,22])
	star_binding_energy = stars_grp.create_dataset('binding_energy', data=stack[:,25])
	halo_mask = stars_grp.create_dataset('halo_mask', data=stack[:,26])
	disc_mask = stars_grp.create_dataset('disc_mask', data=stack[:,27])
	f.close()
	
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

class Snapshot(object):
	def __init__(self, run=default_run,tag=default_tag,model=default_model,directory=default_dir, mass_cut=[2e10,7e10]):
		self.run = run
		self.tag = tag
		self.model = model
		self.directory = directory
		self.mass_cut = mass_cut
		self.particlestack = []
		self.fofstack = []
		self.simattributes = []
	
	def load(self):
		 self.particlestack, self.fofstack, self.simattributes = loadparticles(run = self.run, tag = self.tag, model = self.model, directory = self.directory, mass_cut=self.mass_cut)	

class Snipshot(object):
	def __init__(self, groupnum, run=default_run,tag=default_tag,model=default_model,directory=default_dir):
		self.groupnum = groupnum
		self.run = run
		self.tag = tag
		self.model = model
		self.directory = directory
		self.particlestack = []
		self.fofstack = []
		self.simattributes = []
	def load(self):
		 self.particlestack, self.fofstack, self.simattributes = loadsnipparticles(self.groupnum, run = self.run, tag = self.tag, model = self.model, directory = self.directory)

class Halo(object):
	def __init__(self, groupnumber, snip=False):
		self.snip = snip	
		self.groupnumber = groupnumber
		self.particles = []
		self.fof = []

	def loadfromfile(self, run,model,tag,directory = work_dir):
		if self.snip == True:
			print 'Cannot Load Snip from file!'
			return None
		subfolder = 'savedhalos/%s/%s/%s/' %(run, model, tag)
		filename = str(directory)+str(subfolder)+str(run)+'_'+str(model)+'_'+str(tag)+'_FOF'+str(int(self.groupnumber))+'.hdf5'
		if not os.path.exists(filename):
			raise RuntimeError('FILE NOT FOUND: '+str(filename))
		f = h5py.File(filename, 'r')
		sa = f['/simattributes']
		fof = f['/fofdata']
		st = f['/stardata']
		self.simattributes = [sa['run'].value, sa['model'].value, sa['tag'].value, sa['h'].value, sa['boxsize'].value, sa['masstable'].value, sa['mass_cut'].value]
		self.fof = np.array([fof['groupnum'].value, fof['stellarmass'].value, fof['stellar_fe_h'].value, fof['low_high_a_fe_ratio'].value, fof['percenthigh_a_fe'].value, fof['ratio_jz_jc_sim1'].value, fof['fe_h_gradient'].value, fof['r200'].value, fof['scalelength'].value, fof['sersicindex'].value])
		abunds = st['abundances'][:,:]
		self.particles = np.dstack((st['type'][:], st['position'][:,0], st['position'][:,1], st['position'][:,2], st['velocity'][:,0], st['velocity'][:,1], st['velocity'][:,2], st['mass'][:], st['fe_h'][:], st['a_fe'][:], st['jz'][:], st['jc'][:], st['jz_jc'][:], abunds[:,0], abunds[:,1], abunds[:,2], abunds[:,3], abunds[:,4], abunds[:,5], abunds[:,6], abunds[:,7], abunds[:,8], st['age'][:]))[0]
		
	def loadfromsim(self, simobject):
		self.simattributes = simobject.simattributes
		self.particles, self.fof = halo(simobject.particlestack, simobject.fofstack, self.simattributes, self.groupnumber, snip=self.snip)		

	def image(self, my_clim = (0,150), set_clim=False, save = False):

		x = self.particles[:,1]
		y = self.particles[:,2]
		z = self.particles[:,3]
		starmass = self.particles[:,7]
		params = {'axes.labelsize': 14, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'text.usetex': True, 'lines.linewidth' : 2}
		plt.rcParams.update(params)
		from mpl_toolkits.mplot3d import Axes3D
		fig = plt.figure()
		my_cmap = matplotlib.cm.get_cmap('jet')
		my_cmap.set_under('w')
		ybins = np.arange(-0.06, 0.06, 0.001)
		xbins = np.arange(-0.06, 0.06, 0.001)
		hist, bins = np.histogram(np.abs(x), bins=100, range=(0,0.06))
		hist = np.log10(hist)
		centers = np.log10((bins[:-1] + bins[1:]) / 2)
		gs = gridspec.GridSpec(2,2)
		ax1 = fig.add_subplot(gs[0,0])
		ax1.scatter(centers,hist, s=3)
		ax2 = fig.add_subplot(gs[0,1])
		H, xedges, yedges = np.histogram2d(y,x,bins=(ybins,xbins))
		im2 = ax2.imshow(H, extent=[-0.06,0.06,-0.06,0.06], interpolation='nearest', origin='lower', aspect='auto',cmap=my_cmap, vmin=0.1, clim=my_clim)
		#ax2.scatter(x,y, s=starmass+10)
		ax2.set_xlabel(r'$x[Mpc]$')
		ax2.set_ylabel(r'$y[Mpc]$')
		ax3 = fig.add_subplot(gs[1,0])
		H, xedges, yedges = np.histogram2d(z,x,bins=(ybins,xbins))
		im3 = ax3.imshow(H, extent=[-0.06,0.06,-0.06,0.06], interpolation='nearest', origin='lower', aspect='auto',cmap=my_cmap, vmin=0.1, clim=my_clim)
		#ax3.scatter(x,z, s=starmass+10)
		ax3.set_xlabel(r'$x[Mpc]$')
		ax3.set_ylabel(r'$z[Mpc]$')
		ax4 = fig.add_subplot(gs[1,1])
		H, xedges, yedges = np.histogram2d(z,y,bins=(ybins,xbins))
		im4 = ax4.imshow(H, extent=[-0.06,0.06,-0.06,0.06], interpolation='nearest', origin='lower', aspect='auto',cmap=my_cmap, vmin=0.1, clim=my_clim)
		#ax4.scatter(y,z, s=starmass+10)
		ax4.set_xlabel(r'$y[Mpc]$')
		ax4.set_ylabel(r'$z[Mpc]$')
		fig.subplots_adjust(hspace=0.30, wspace=0.30, right=0.95, top=0.95, bottom=0.10)
		fof = "FOF "+ str(int(self.groupnumber))
		fig.text(0.5, 0.96, fof)
		if set_clim == True:
			axcolor = 'lightgoldenrodyellow'
			axmin = fig.add_axes([0.05, 0.02, 0.9, 0.02], axisbg=axcolor)
			axmax  = fig.add_axes([0.05, 0.05, 0.9, 0.02], axisbg=axcolor) 
			smin = Slider(axmin, 'Min', 0, len(self.particles[:,1]), valinit=0)
			smax = Slider(axmax, 'Max', 0, len(self.particles[:,1]), valinit=150)
			def update(val):
    				im2.set_clim([smin.val,smax.val])
				im3.set_clim([smin.val,smax.val])
				im4.set_clim([smin.val,smax.val])
    				fig.canvas.draw()
			smin.on_changed(update)
			smax.on_changed(update)


		if save == True:
			plttitle = plot_dir+self.simattributes[0]+'_'+self.simattributes[1]+'/' + 'FOF'+str(int(self.groupnumber))+'pos.png'
			plt.savefig(plttitle, format='png', dpi = 1200)
			plt.close(fig)
		if save != True:
			plt.show()
	
	def jz_jc_distribution(self, save = False):
		hist, bins = np.histogram(self.particles[:,12], bins=100, range=(-2,2))
		centers = (bins[:-1] + bins[1:]) / 2
		hist = [float(histo)/float(sum(hist)) for histo in hist]
		xbins = np.arange(0,0.05,0.001)
		ybins = np.arange(-2,2,0.05)
		H, xedges, yedges = np.histogram2d(self.particles[:,12],np.linalg.norm(self.particles[:,1:3], axis=1),bins=(ybins,xbins))
		my_cmap = matplotlib.cm.get_cmap('jet')
		my_cmap.set_under('w')
		params = {'axes.labelsize': 14, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'text.usetex': True, 'lines.linewidth' : 2}
		plt.rcParams.update(params)
		fig, ax = plt.subplots(1,2)
		ax[0].plot(centers,hist)
		ax[0].set_ylabel(r'$N/N*$')
		ax[0].set_xlabel(r'$j_z / j_c$')
		ax[0].set_xlim(-2,2)
		ax[1].imshow(H, extent =[0,0.05,-2,2], interpolation='nearest', origin='lower', aspect='auto',cmap=my_cmap, vmin=0.1)
		ax[1].set_xlabel(r'$R$ $[Mpc]$')
		ax[1].set_ylabel(r'$j_z / j_c$')
		if save == True:
			plttitle = plot_dir+self.simattributes[0]+'_'+self.simattributes[1]+'/' + 'FOF'+str(int(self.groupnumber))+'jzjc.png'
			plt.savefig(plttitle, format='png', dpi = 1200)
			plt.close(fig)
		if save != True:
			plt.show()
	
	def alphafe(self, rad_low=0.003, rad_high=0.015, z_low=0.00, z_high=0.003, save = False):
		if self.snip==True:
			print 'Alpha elements not defined for snipshot!'
			return None
		R_pos = np.linalg.norm(self.particles[:,1:3], axis =1)
		z_pos = self.particles[:,3]
		a_fe = self.particles[:,9]
		fe_h = self.particles[:,8]
		mass = self.particles[:,7]
		r_200 = self.fof[7]
		rad_bins = np.linspace(rad_low,rad_high,num=7, endpoint=True)
		z_bins = np.linspace(z_low,z_high,num=4, endpoint=True)
		#rad_bins = np.linspace(0.0,0.1*r_200,num=7, endpoint=False)
		#z_bins = np.linspace(0.00,0.1*r_200,num=4, endpoint=False)
		#bin data into 6 radial and height bins (as in Hayden et al. 2015)
		radial_fe_h = []
		radial_a_fe = []
		radial_mass = []
		for j in range(0,len(z_bins)-1):
			z_mask = [(np.abs(z_pos) > z_bins[j]) & (np.abs(z_pos) < z_bins[j+1])]
			fe_h_zbin = fe_h[z_mask]
			a_fe_zbin = a_fe[z_mask]
			r_zbin = R_pos[z_mask]
			mass_zbin = mass[z_mask]
			for i in range(0,len(rad_bins)-1):
				bin_mask = [(np.abs(r_zbin) > rad_bins[i]) & (np.abs(r_zbin) < rad_bins[i+1])]
				fe_h_bin = fe_h_zbin[bin_mask]
				a_fe_bin = a_fe_zbin[bin_mask]
				mass_bin = mass_zbin[bin_mask]
				radial_fe_h.append(fe_h_bin)
				radial_a_fe.append(a_fe_bin)
				radial_mass.append(mass_bin)
		xbins = np.linspace(-2, 1, 40)
		ybins = np.linspace(-0.1,0.6, 40)
		my_cmap = matplotlib.cm.get_cmap('jet')
		my_cmap.set_under('w')
		#produce plot
		params = {'axes.labelsize': 14, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'text.usetex': True, 'lines.linewidth' : 2, 'axes.titlesize' : 6}
		plt.rcParams.update(params)
		f, ((ax1,ax2,ax3,ax4,ax5,ax6),(ax7,ax8,ax9,ax10,ax11,ax12),(ax13,ax14,ax15,ax16,ax17,ax18)) = plt.subplots(3,6,sharey='row', sharex='row')
		my_extent = [-2,1,-0.1,0.6]
		aspect = 'auto'
		H, xedges, yedges = np.histogram2d(radial_a_fe[0],radial_fe_h[0],bins=(ybins,xbins), weights=radial_mass[0])
		ax13.imshow(H, extent=my_extent, interpolation='nearest', origin='lower',aspect=aspect, cmap=my_cmap, vmin=0.1)
		H, xedges, yedges = np.histogram2d(radial_a_fe[1],radial_fe_h[1],bins=(ybins,xbins), weights=radial_mass[1])
		ax14.imshow(H, extent=my_extent, interpolation='nearest', origin='lower',aspect=aspect, cmap=my_cmap, vmin=0.1)
		H, xedges, yedges = np.histogram2d(radial_a_fe[2],radial_fe_h[2],bins=(ybins,xbins), weights=radial_mass[2])
		ax15.imshow(H, extent=my_extent, interpolation='nearest', origin='lower',aspect=aspect, cmap=my_cmap, vmin=0.1)
		H, xedges, yedges = np.histogram2d(radial_a_fe[3],radial_fe_h[3],bins=(ybins,xbins), weights=radial_mass[3])
		ax16.imshow(H, extent=my_extent, interpolation='nearest', origin='lower',aspect=aspect, cmap=my_cmap, vmin=0.1)
		H, xedges, yedges = np.histogram2d(radial_a_fe[4],radial_fe_h[4],bins=(ybins,xbins), weights=radial_mass[4])
		ax17.imshow(H, extent=my_extent, interpolation='nearest', origin='lower',aspect=aspect, cmap=my_cmap, vmin=0.1)
		H, xedges, yedges = np.histogram2d(radial_a_fe[5],radial_fe_h[5],bins=(ybins,xbins), weights=radial_mass[5])
		ax18.imshow(H, extent=my_extent, interpolation='nearest', origin='lower',aspect=aspect, cmap=my_cmap, vmin=0.1)
		H, xedges, yedges = np.histogram2d(radial_a_fe[6],radial_fe_h[6],bins=(ybins,xbins), weights=radial_mass[6])
		ax7.imshow(H, extent=my_extent, interpolation='nearest', origin='lower',aspect=aspect, cmap=my_cmap, vmin=0.1)
		H, xedges, yedges = np.histogram2d(radial_a_fe[7],radial_fe_h[7],bins=(ybins,xbins), weights=radial_mass[7])
		ax8.imshow(H, extent=my_extent, interpolation='nearest', origin='lower',aspect=aspect, cmap=my_cmap, vmin=0.1)
		H, xedges, yedges = np.histogram2d(radial_a_fe[8],radial_fe_h[8],bins=(ybins,xbins), weights=radial_mass[8])
		ax9.imshow(H, extent=my_extent, interpolation='nearest', origin='lower',aspect=aspect, cmap=my_cmap, vmin=0.1)
		H, xedges, yedges = np.histogram2d(radial_a_fe[9],radial_fe_h[9],bins=(ybins,xbins), weights=radial_mass[9])
		ax10.imshow(H, extent=my_extent, interpolation='nearest', origin='lower',aspect=aspect, cmap=my_cmap, vmin=0.1)
		H, xedges, yedges = np.histogram2d(radial_a_fe[10],radial_fe_h[10],bins=(ybins,xbins), weights=radial_mass[10])
		ax11.imshow(H, extent=my_extent, interpolation='nearest', origin='lower',aspect=aspect, cmap=my_cmap, vmin=0.1)
		H, xedges, yedges = np.histogram2d(radial_a_fe[11],radial_fe_h[11],bins=(ybins,xbins), weights=radial_mass[11])
		ax12.imshow(H, extent=my_extent, interpolation='nearest', origin='lower',aspect=aspect, cmap=my_cmap, vmin=0.1)
		H, xedges, yedges = np.histogram2d(radial_a_fe[12],radial_fe_h[12],bins=(ybins,xbins), weights=radial_mass[12])
		ax1.imshow(H, extent=my_extent, interpolation='nearest', origin='lower',aspect=aspect, cmap=my_cmap, vmin=0.1)
		H, xedges, yedges = np.histogram2d(radial_a_fe[13],radial_fe_h[13],bins=(ybins,xbins), weights=radial_mass[13])
		ax2.imshow(H, extent=my_extent, interpolation='nearest', origin='lower',aspect=aspect, cmap=my_cmap, vmin=0.1)
		H, xedges, yedges = np.histogram2d(radial_a_fe[14],radial_fe_h[14],bins=(ybins,xbins), weights=radial_mass[14])
		ax3.imshow(H, extent=my_extent, interpolation='nearest', origin='lower',aspect=aspect, cmap=my_cmap, vmin=0.1)
		H, xedges, yedges = np.histogram2d(radial_a_fe[15],radial_fe_h[15],bins=(ybins,xbins), weights=radial_mass[15])
		ax4.imshow(H, extent=my_extent, interpolation='nearest', origin='lower',aspect=aspect, cmap=my_cmap, vmin=0.1)	
		H, xedges, yedges = np.histogram2d(radial_a_fe[16],radial_fe_h[16],bins=(ybins,xbins), weights=radial_mass[16])
		ax5.imshow(H, extent=my_extent, interpolation='nearest', origin='lower',aspect=aspect, cmap=my_cmap, vmin=0.1)
		H, xedges, yedges = np.histogram2d(radial_a_fe[17],radial_fe_h[17],bins=(ybins,xbins), weights=radial_mass[17])
		ax6.imshow(H, extent=my_extent, interpolation='nearest', origin='lower',aspect=aspect, cmap=my_cmap, vmin=0.1)
		ax1.set_ylabel(r'$[\alpha/Fe]$')
		ax7.set_ylabel(r'$[\alpha/Fe]$')
		ax13.set_ylabel(r'$[\alpha/Fe]$')
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
		'''
		for ax in axes:
			ax.set_ylim([-0.1,0.6])
			ax.set_xlim([-1.5,0.5])
			ax.yaxis.set_ticks(np.arange(-0.6,1.2,0.2))
			ax.xaxis.set_ticks(np.arange(-2.0,2.0,1.0))
		'''
		f.subplots_adjust(hspace=0.40, wspace=0, left=0.08, right=0.97, top=0.92, bottom=0.08)
		f.text(0.45, 0.34, r'$'+str(z_bins[0]*1000)+' < |z| < '+str(z_bins[1]*1000)+'$ $Kpc$', fontsize=9)
		f.text(0.45, 0.65, r'$'+str(z_bins[1]*1000)+' < |z| < '+str(z_bins[2]*1000)+'$ $Kpc$', fontsize=9)
		f.text(0.45, 0.96, r'$'+str(z_bins[2]*1000)+' < |z| < '+str(z_bins[3]*1000)+'$ $Kpc$', fontsize=9)
		#f.set_size_inches(12,6, forward=True)
	
		if save == True:
			plttitle = plot_dir+self.simattributes[0]+'_'+self.simattributes[1]+'/' + 'FOF'+str(int(self.groupnumber))+'alphafe.png'
			plt.savefig(plttitle, format='png', dpi = 1200)
			plt.close(f)
		if save != True:
			plt.show()

	def mdf(self, rad_low=0.003, rad_high=0.015, z_low=0.00, z_high=0.003,  save = False):
		if self.snip == True:
			print 'Fe/H is not well defined for snipshot!'
			return None
		R_pos = np.linalg.norm(self.particles[:,1:3], axis=1)
		z_pos = self.particles[:,3]
		a_fe = self.particles[:,9]
		fe_h = self.particles[:,8]
		mass = self.particles[:,7]
		r_200 = self.fof[7]
		rad_bins = np.linspace(rad_low,rad_high,num=7, endpoint=True)
		z_bins = np.linspace(z_low,z_high,num=4, endpoint=True)
		radial_fe_h = []
		radial_mass = []
		radial_a_fe = []
		for j in range(0,len(z_bins)-1):
			z_mask = [(np.abs(z_pos) > z_bins[j]) & (np.abs(z_pos) < z_bins[j+1])]
			fe_h_zbin = fe_h[z_mask]
			mass_zbin = mass[z_mask]
			a_fe_zbin = a_fe[z_mask]
			r_zbin = R_pos[z_mask]
			for i in range(0,len(rad_bins)-1):
				bin_mask = [(np.abs(r_zbin) > rad_bins[i]) & (np.abs(r_zbin) < rad_bins[i+1])]
				fe_h_bin = fe_h_zbin[bin_mask]
				a_fe_bin = a_fe_zbin[bin_mask]
				mass_bin = mass_zbin[bin_mask]
				radial_fe_h.append(fe_h_bin)
				radial_a_fe.append(a_fe_bin)
				radial_mass.append(mass_bin)
		len_zbins = len(radial_fe_h)/len(z_bins)
		fe_h_hists = []
		fe_h_centers = []
		for i in range(0,len(radial_fe_h)):
			hist, bins = np.histogram((radial_fe_h[i]), bins=15, weights=radial_mass[i], range=(-3,1.0))
			rad_feh = radial_fe_h[i]
			centers = ((bins[:-1] + bins[1:]) / 2)
			hist = [float(histo)/float(sum(hist)) for histo in hist]
			fe_h_hists.append(hist)
			fe_h_centers.append(centers)
	
		labels = [r'$'+str(rad_bins[0]*1000)+' < R < '+str(rad_bins[1]*1000)+'$ $Kpc$', r'$'+str(rad_bins[1]*1000)+' < R < '+str(rad_bins[2]*1000)+'$ $Kpc$', r'$'+str(rad_bins[2]*1000)+' < R < '+str(rad_bins[3]*1000)+'$ 	$Kpc$', r'$'+str(rad_bins[3]*1000)+' < R < '+str(rad_bins[4]*1000)+'$ $Kpc$', r'$'+str(rad_bins[4]*1000)+' < R < '+str(rad_bins[5]*1000)+'$ $Kpc$', r'$'+str(rad_bins[5]*1000)+' < R < '+str(rad_bins[6]*1000)+'$ $Kpc$']
	
		params = {'axes.labelsize': 14, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'text.usetex': True, 'lines.linewidth' : 2}
		plt.rcParams.update(params)
		fig, ax = plt.subplots(3,1, sharex=True)
		
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
		if save == True:
			plttitle = plot_dir+self.simattributes[0]+'_'+self.simattributes[1]+'/' + 'FOF'+str(int(self.groupnumber))+'mdf.png'
			plt.savefig(plttitle, format='png', dpi = 1200)
			plt.close(fig)
		if save != True :
			plt.show()
	
	def agescaleheight(self, rad_cut = [0.003,0.015], jzjc_cut=[0.8,1.2], save=False, returnarray=False):
		if self.snip == True:
			print 'You didnt do ages for the snipshots yet...'
			return None
		part = self.particles
		jzjc = part[:,12]
		R_pos = np.linalg.norm(part[:,1:3], axis=1)
		disk = (jzjc < jzjc_cut[1]) & (jzjc > jzjc_cut[0]) & (R_pos < rad_cut[1]) & (R_pos > rad_cut[0])
		z_disk = part[:,3][disk]
		age_disk = part[:,22][disk]
		mass_disk = part[:,7][disk]
		def exponential(z, N_0, z_s, n):
			return N_0 * np.exp(-1*(z/z_s)**n)
		hist, bins = np.histogram(np.abs(z_disk), bins=100, range=(0,0.003))
    		centers = (bins[:-1] + bins[1:]) / 2
		global_popt, global_pcov = curve_fit(exponential, centers, hist, p0=[hist[0],0.002,1])
		global_scale_height = global_popt[1]
		age_bins = np.linspace(min(age_disk), max(age_disk), num=15, endpoint=True)
		scale_heights = []
		bin_center = []
		for i in range(0,len(age_bins)-1):
			age_mask = (age_disk < age_bins[i+1]) & (age_disk > age_bins[i])
   			z_age = z_disk[age_mask]
			mass_age = mass_disk[age_mask]
    			hist, bins = np.histogram(np.abs(z_age), bins=100, range=(0,0.003),weights=mass_age)
    			centers = (bins[:-1] + bins[1:]) / 2
			try:
    				popt, pcov = curve_fit(exponential, centers, hist, p0=[hist[0],0.002,1])
			except RuntimeError:
				print 'WARNING: could not fit scale height for halo '+ str(self.groupnumber)+'... check disk alignment?'
				return None
    			bin = age_bins[i]+((age_bins[i+1]-age_bins[i])/2)
    			bin_center.append(bin)
    			scale_heights.append(popt[1])
		scale_heights = np.array(scale_heights)*1e3
   		plt.plot(bin_center, scale_heights)
		plt.xlim(bin_center[0]-0.2, bin_center[-1]+0.2)
		#plt.ylim(scale_heights[0]-0.1*scale_heights[0], scale_heights[-1]+0.05*scale_heights[-1])weights=mass_age
		plt.xlabel(r'$Age$ $[Gyr]$')
		plt.ylabel(r'$Scale$ $Height$ $[Kpc]$')
		if save == True:
			plttitle = plot_dir+self.simattributes[0]+'_'+self.simattributes[1]+'/' + 'FOF'+str(int(self.groupnumber))+'diskageheight.png'
			plt.savefig(plttitle, format='png', dpi = 1200)
			plt.close()
		if save != True and returnarray == False:
			plt.show()
		if returnarray==True:
			return bin_center, scale_heights
			

	def save(self, directory=work_dir):
		if self.snip == True:
			print 'You probably dont want to save snipshots... or do you?'
			return None
		savehaloarrays(self.particles, self.fof, self.simattributes, directory=directory)
	


	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
