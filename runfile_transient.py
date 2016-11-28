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
total_T1 = 10e6  #yrs. 
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
uplift_per_step = uplift_rate * dt

##eroder/diffuser
Ksp1 = 1e-5
Ksp2 = 1e-3
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

#Main Loop 1 (After first sucess is confirmed this is all moved in a class....)
t0 = time.time()
while elapsed_time < total_T1:
    #mg.at_node['topographic__slope']  = mg.calc_slope_at_node(z) #Calculate slope for the detachment-eroder
    fr.route_flow()
    ld.run_one_step(dt=dt)
    fc.run_one_step(dt=dt)
    z[mg.core_nodes] += uplift_rate * dt #add uplift
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
        write_netcdf('./NC/output{}'.format(elapsed_time)+'__'+str(int(elapsed_time/oi)).zfill(zp)+'.nc',
                mg,format='NETCDF4')
    elapsed_time += dt #update elapsed time
tE = time.time()
print()
print('End of  Main Loop. So far it took {}s to get here. No worries homeboy...'.format(tE-t0))

