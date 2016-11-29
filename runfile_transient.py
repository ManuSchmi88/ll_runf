#Import necessesary Python and Landlab Modules
import numpy as np
from landlab import RasterModelGrid
from landlab import CLOSED_BOUNDARY, FIXED_VALUE_BOUNDARY
from landlab.components import FlowRouter
from landlab.components import LinearDiffuser
from landlab.components import FastscapeEroder
from landlab import imshow_grid
from landlab.io.netcdf import write_netcdf
from landlab.io.netcdf import read_netcdf
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import time

#Build up Grid and define variables

##DOMAIN
##This is also not needed if you use NetCDF input.
#ncols = 601 
#nrows = 601 
#dx    = 100

##TIME
##runtime
##Important to know: total_T2 is the TOTAL RUNTIME of the model
##After total_T1 there is a change in parameters.
total_T1 = 5e5  #50.000 yrs. 
total_T2 = 3e6  #3.000.000 yrs
##timestep
dt = 100        #yrs
##number of timesteps and fillfactor for ffmpeg printout
nt = total_T1/dt
##Set the interval at which we create output. Warning. High output frequency
##will slow down the script significantly
oi = 5000
no = total_T1/oi
zp = len(str(int(no)))
##uplift
uplift_rate = 5e-4 #m/yr

##eroder/diffuser
Ksp1 = 1e-5
Ksp2 = 5e-4 
msp  = 0.5
nsp  = 1.0
ldi   = 1e-2
ldi2  = 1e-1
#time
elapsed_time = 0

##INITIALIZE LANDLAB COMPONENTGRID
##Uncomment/Comment regarding if you are using NetCDF input
#NetCDF file should be provided within ./ directory
mg = read_netcdf('Nc_in.nc')

##Create everything new, without netcdf input
#mg = RasterModelGrid((nrows,ncols),dx)
#z  = mg.add_ones('node','topographic__elevation')
#ir = np.random.rand(z.size)/1000 #builds up initial roughness
#z += ir #adds initial roughness to grid

#SET UP BOUNDARY CONDITIONS
for edge in (mg.nodes_at_left_edge,mg.nodes_at_right_edge):
    mg.status_at_node[edge] = CLOSED_BOUNDARY
for edge in (mg.nodes_at_top_edge,mg.nodes_at_bottom_edge):
    mg.status_at_node[edge] = FIXED_VALUE_BOUNDARY

#Create Threshold_sp field
threshold_arr  = mg.zeros('node',dtype=float)
threshold_arr += 0.
threshold_field = mg.add_field('node','threshold_sp',threshold_arr,noclobber = False)
imshow_grid(mg,'threshold_sp')
plt.title('Stream-Power Threshold')
plt.savefig('Distribution of SP_Threshold',dpi=720)

#Initialize the erosional components
fr  = FlowRouter(mg)
ld  = LinearDiffuser(mg,linear_diffusivity=ldi)
fc  = FastscapeEroder(mg,K_sp = Ksp1,m_sp=msp, n_sp=nsp, threshold_sp=threshold_arr)
ld2 = LinearDiffuser(mg,linear_diffusivity=ldi2)
fc2 = FastscapeEroder(mg,K_sp = Ksp2,m_sp=msp, n_sp=nsp, threshold_sp=threshold_arr)


##------------------------------------------------##
##-------START OF THE SECOND LOOP-----------------##
##------------------------------------------------##
##(After first sucess is confirmed this is all moved in a class....)

t0 = time.time() #start system timer

while elapsed_time < total_T1:

    #Erosional routines:
    fr.route_flow()
    ld.run_one_step(dt=dt)
    fc.run_one_step(dt=dt)
    
    #Add uplift
    mg.at_node['topographic__elevation'][mg.core_nodes] += uplift_rate * dt #add uplift

    #Output every oi - years
    if elapsed_time % oi  == 0:
        print('Elapsed Time:' , elapsed_time,'writing output!')
        #Create DEM
        plt.figure()
        imshow_grid(mg,'topographic__elevation',grid_units=['m','m'],var_name = 'Elevation',cmap='terrain')
        plt.savefig('./DEM/DEM_t{}'.format(elapsed_time)+'__'+str(int(elapsed_time/oi)).zfill(zp)+'.png')
        plt.close()
        #Create Flow Accumulation Map
        plt.figure()
        imshow_grid(mg,fr.drainage_area,grid_units=['m','m'],var_name =
        'Drainage Area',cmap='bone')
        plt.savefig('./ACC/ACC_t{}'.format(elapsed_time)+'__'+str(int(elapsed_time/oi)).zfill(zp)+'.png')
        plt.close()
        #Create Slope - Area Map
        plt.figure()
        plt.loglog(mg.at_node['drainage_area'][np.where(mg.at_node['drainage_area'] > 0)],
           mg.at_node['topographic__slope'][np.where(mg.at_node['drainage_area'] > 0)],
           marker='.',linestyle='None')
        plt.xlabel('Area')
        plt.ylabel('Slope')
        plt.savefig('./SA/SA_t{}'.format(elapsed_time)+'__'+str(int(elapsed_time/oi)).zfill(zp)+'.png')
        plt.close()
        #Create NetCDF Output
        write_netcdf('./NC/output_t{}'.format(elapsed_time)+'__'+str(int(elapsed_time/oi)).zfill(zp)+'.nc',
                mg,format='NETCDF4')

    elapsed_time += dt #update elapsed time
tE = time.time()
print()
print('First loop done!')
print('We now switch the to the second part with a lin_diff of {} and and K_sp of {}'.format(ldi2,Ksp2))
print('So far it took {}s to run'.format(tE - t0))

##------------------------------------------------##
##-------START OF THE SECOND LOOP-----------------##
##------------------------------------------------##

while elapsed_time < total_T2:
    
    #Actual erosion routines:
    fr.run_one_step(dt=dt)
    ld2.run_one_step(dt=dt)
    fc2.run_one_step(dt=dt)
    
    #Add uplift
    mg.at_node['topographic__elevation'][mg.core_nodes] += uplift_rate * dt #add uplift

    #Output every oi - years
    if elapsed_time % oi == 0:
        print('Elapsed Time:' , elapsed_time,'writing output!')
        plt.figure()
        imshow_grid(mg,'topographic__elevation',grid_units=['m','m'],var_name = 'Elevation',cmap='terrain')
        plt.savefig('./DEM/DEM_t{}'.format(elapsed_time)+'__'+str(int(elapsed_time/oi)).zfill(zp)+'.png')
        plt.close()
        plt.figure()
        imshow_grid(mg,fr.drainage_area,cmap='bone')
        plt.savefig('./ACC/ACC_t{}'.format(elapsed_time)+'__'+str(int(elapsed_time/oi)).zfill(zp)+'.png')
        plt.close()
        plt.figure()
        plt.loglog(mg.at_node['drainage_area'][np.where(mg.at_node['drainage_area'] > 0)],
           mg.at_node['topographic__slope'][np.where(mg.at_node['drainage_area'] > 0)],
           marker='.',linestyle='None')
        plt.ylim([0.1,1])
        plt.xlabel('Area')
        plt.ylabel('Slope')
        plt.savefig('./SA/SA_t{}'.format(elapsed_time)+'__'+str(int(elapsed_time/oi)).zfill(zp)+'.png')
        plt.close()
        write_netcdf('./NC/output_t{}'.format(elapsed_time)+'__'+str(int(elapsed_time/oi)).zfill(zp)+'.nc'
                     ,mg,format='NETCDF4')
    
    #update elapsed time
    elapsed_time += dt
tE = time.time()
print()
print('Second loop done!')
print('We are finished will all loops. Can I go sleep now?')
