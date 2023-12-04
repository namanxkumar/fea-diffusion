from accuracy import calculate_accuracy
import os
import numpy as np

steps = 3
samples = 50
conditions_per_plate = 1

def accuracy(steps, samples, conditions_per_plate):
    loss_values = []
    for i in range(1, samples+1):
        for j in range(1, conditions_per_plate+1):
            for k in range(1, steps):    
                mesh_path = "/BTP/fea_diffusion/accuracydatagen/"+str(i)+"/"+str(j)+"/"+"domain."+str(k)+".vtk"
                x_displacement_path = "/BTP/fea_diffusion/accuracygeneratedresults/test/"+str(i)+"/"+str(j)+"/"+"sample-1.png"
                y_displacement_path = "/BTP/fea_diffusion/accuracygeneratedresults/test/"+str(i)+"/"+str(j)+"/"+"sample-2.png"
                loss_values.append(calculate_accuracy(mesh_path,x_displacement_path, y_displacement_path))

    return loss_values

loss_values = np.array(accuracy(steps, samples, conditions_per_plate))
print(loss_values)
print("Mean: ", np.mean(loss_values))