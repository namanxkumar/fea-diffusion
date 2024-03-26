import pyvista as pv
import numpy as np



max_displacement = np.zeros(1000)
min_displacement = np.zeros(1000)
mean_displacement = np.zeros(1000)
median_displacement = np.zeros(1000)
std_displacement = np.zeros(1000)
n = 0
for i in range(1,1001):
    for j in range(1,5):
        mesh = pv.read(f'data/feadata/{i}/{j}/domain.5.vtk/')
        displacement = np.array(mesh.point_data["u"][:,:2])
        if (displacement <= 0).all():
            n += 1
            # if (k > 10**13).any() or (k < -10**13).any():
            #     print(k)
           
                
        
        displacement = np.abs(displacement)
        max_displacement[i-1] = (np.max(displacement))
        
        min_displacement[i-1] = np.min(displacement[displacement > 0]) if displacement[displacement > 0].size else np.nan

        mean_displacement[i-1] =(np.mean(displacement))
        median_displacement[i-1] =(np.median(displacement))
        std_displacement[i-1] =(np.std(displacement))
     
# print(max_displacement)
print(n)
print("Max displacement: ", np.nanmean(max_displacement))
print("Min displacement: ", np.nanmean(min_displacement))
print("Mean displacement: ", np.nanmean(mean_displacement))
print("Median displacement: ", np.nanmean(median_displacement))
print("Std displacement: ", np.nanmean(std_displacement))


# mesh = pv.read(f'data/feadata/2/2/domain.1.vtk/')
# displacement = np.array(mesh.point_data["u"][:,:2])
# displacement = np.abs(displacement)
# print(displacement)
# print(displacement[np.where(displacement != 0)])