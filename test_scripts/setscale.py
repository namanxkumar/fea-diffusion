import pyvista as pv
import numpy as np



max_displacement = np.zeros(40)
min_displacement = np.zeros(40)
mean_displacement = np.zeros(40)
median_displacement = np.zeros(40)
std_displacement = np.zeros(40)

# for i in range(1,40):
#     mesh = pv.read(f'data/sanitycheck/largeforce/{i}/1/domain.1.vtk/')
#     displacement = np.array(mesh.point_data["u"][:,:2])
#     displacement = np.abs(displacement)
#     print(i, displacement)
#     max_displacement[i-1] = (np.max(displacement))
    
#     min_displacement[i-1] = np.min(displacement[displacement > 0]) if displacement[displacement > 0].size else np.nan

#     mean_displacement[i-1] =(np.mean(displacement))
#     median_displacement[i-1] =(np.median(displacement))
#     std_displacement[i-1] =(np.std(displacement))
     

# print("Max displacement: ", np.nanmean(max_displacement))
# print("Min displacement: ", np.nanmean(min_displacement))
# print("Mean displacement: ", np.nanmean(mean_displacement))
# print("Median displacement: ", np.nanmean(median_displacement))
# print("Std displacement: ", np.nanmean(std_displacement))


mesh = pv.read(f'data/feadata/2/2/domain.1.vtk/')
displacement = np.array(mesh.point_data["u"][:,:2])
displacement = np.abs(displacement)
print(displacement)
print(displacement[np.where(displacement != 0)])