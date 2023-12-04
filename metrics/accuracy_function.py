import numpy as np
import pyvista as pv
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


def calculate_accuracy_for_one_sample(
    mesh_file, displacement_x_file, displacement_y_file, image_size
):
    mesh = pv.read(mesh_file)
    cords = np.array(mesh.points)[:, :2]
    x_max = np.max(cords[:, 0])
    y_max = np.max(cords[:, 1])

    if x_max <= y_max:
        cords = np.stack((cords[:, 0] + (1 - (x_max)) / 2, cords[:, 1]), axis=1)
    else:
        cords = np.stack((cords[:, 0], cords[:, 1] + (1 - (y_max)) / 2), axis=1)

    cords = cords * image_size

    ground_truth_displacement = np.array(mesh.point_data["u"])[:, :2]

    pixels_min = np.floor(cords)
    pixels_min = pixels_min.astype(int)
    pixels_max = np.ceil(cords)
    pixels_max = pixels_max.astype(int)

    predicted_displacement_x = np.array(
        # ImageOps.grayscale(Image.open("outputs_displacement_x_10.png").resize((256, 256)).transpose(Image.FLIP_TOP_BOTTOM))
        ImageOps.grayscale(
            Image.open(displacement_x_file)
            .resize((image_size, image_size))
            .transpose(Image.ROTATE_270)
        )
        # ImageOps.grayscale(Image.open("outputs_displacement_x_10.png").resize((256, 256)))
    ).squeeze()
    predicted_displacement_y = np.array(
        # ImageOps.grayscale(Image.open("outputs_displacement_y_10.png").resize((256, 256)).transpose(Image.FLIP_TOP_BOTTOM))
        ImageOps.grayscale(
            Image.open(displacement_y_file)
            .resize((image_size, image_size))
            .transpose(Image.ROTATE_270)
        )
        # ImageOps.grayscale(Image.open("outputs_displacement_y_10.png").resize((256, 256)))
    ).squeeze()
    # geometry = np.array(
    #     ImageOps.grayscale(
    #         Image.open("input.png").resize((256, 256)).transpose(Image.ROTATE_270)
    #     )
    # ).squeeze()
    # geometry = geometry / 255.0
    # geometry = np.stack((geometry, geometry), axis=0)
    predicted_displacement = np.stack(
        (predicted_displacement_x, predicted_displacement_y), axis=0
    )
    predicted_displacement = predicted_displacement / 255.0
    predicted_displacement = ((predicted_displacement * 2) - 1) * 0.05

    x1 = pixels_min[:, 0]
    y1 = pixels_min[:, 1]
    x2 = pixels_max[:, 0]
    y2 = pixels_max[:, 1]

    # locations = np.argwhere(geometry[x1 - 1, y1 - 1] == 0)
    # print(locations)
    # x1[locations] = x2[locations]
    # y1[locations] = y2[locations]
    # locations = np.argwhere(geometry[x2 - 1, y2 - 1] == 0)
    # print(locations)
    # x2[locations] = x1[locations]
    # y2[locations] = y1[locations]
    x = cords[:, 0]
    y = cords[:, 1]
    q11 = predicted_displacement[:, x1 - 1, y1 - 1]
    q12 = predicted_displacement[:, x1 - 1, y2 - 1]
    q21 = predicted_displacement[:, x2 - 1, y1 - 1]
    q22 = predicted_displacement[:, x2 - 1, y2 - 1]

    x = np.repeat(x[np.newaxis, ...], 2, axis=0)
    y = np.repeat(y[np.newaxis, ...], 2, axis=0)
    x1 = np.repeat(x1[np.newaxis, ...], 2, axis=0)
    y1 = np.repeat(y1[np.newaxis, ...], 2, axis=0)
    x2 = np.repeat(x2[np.newaxis, ...], 2, axis=0)
    y2 = np.repeat(y2[np.newaxis, ...], 2, axis=0)

    l1_loss = lambda y_predicted, y: np.mean(np.abs(y_predicted - y))
    # print(geometry.shape, x1.shape)

    def bilinear_interpolation(x1, x2, y1, y2, x, y, q11, q12, q21, q22):
        np.seterr(invalid="ignore")
        f_xy1 = ((x2 - x) / (x2 - x1)) * q11 + ((x - x1) / (x2 - x1)) * q21

        locations = np.argwhere(np.isnan(f_xy1))
        f_xy1[locations[:, 0], locations[:, 1]] = q11[locations[:, 0], locations[:, 1]]

        # locations = np.argwhere(geometry[:, x1 - 1, y1 - 1] == 0)[:, :2]
        # print(locations.shape, f_xy1.shape, q21.shape)
        # f_xy1[locations] = q21[locations]
        # locations = np.argwhere(geometry[:, x2 - 1, y1 - 1] == 0)[:, :2]
        # f_xy1[locations] = q11[locations]

        f_xy2 = ((x2 - x) / (x2 - x1)) * q12 + ((x - x1) / (x2 - x1)) * q22

        locations = np.argwhere(np.isnan(f_xy2))
        f_xy2[locations[:, 0], locations[:, 1]] = q22[locations[:, 0], locations[:, 1]]

        # locations = np.argwhere(geometry[:, x1 - 1, y2 - 1] == 0)[:, :2]
        # f_xy2[locations] = q22[locations]
        # locations = np.argwhere(geometry[:, x2 - 1, y2 - 1] == 0)[:, :2]
        # f_xy2[locations] = q12[locations]

        f_xy = ((y2 - y) / (y2 - y1)) * f_xy1 + ((y - y1) / (y2 - y1)) * f_xy2

        locations = np.argwhere(np.isnan(f_xy))
        f_xy[locations[:, 0], locations[:, 1]] = f_xy1[locations[:, 0], locations[:, 1]]

        return f_xy.T
        # a = np.repeat(np.expand_dims(1/((x2-x1)*(y2-y1)), axis = 2), 2, axis = 2) * np.stack((x2-x, x-x1), axis = 2)
        # a = np.expand_dims(a, axis = 2)
        # b = np.stack((np.stack((q11, q21), axis = 2 ), np.stack((q12, q22), axis = 2 )), axis = 3)

        # c = np.stack((y2-y, y-y1), axis = 2)
        # c = np.expand_dims(c, axis = -1)
        # print(a.shape, b.shape, c.shape)

        # return np.einsum('CBij,CBji->CBi', (np.einsum('CBij,CBjk->CBik',a,b)), c)

    y_predicted = bilinear_interpolation(x1, x2, y1, y2, x, y, q11, q12, q21, q22)

    ground_truth_displacement = np.clip(ground_truth_displacement, -0.05, 0.05)
    # print(y_predicted)
    # print(y_predicted.shape, ground_truth_displacement.shape)
    # cords = np.delete(cords, np.where(ground_truth_displacement > 0.05))
    # cords = np.reshape(cords, (-1,2))
    # y_predicted = np.delete(y_predicted, np.where(ground_truth_displacement > 0.05))
    # y_predicted = np.reshape(y_predicted, (-1,2))
    # ground_truth_displacement = np.delete(ground_truth_displacement, np.where(ground_truth_displacement > 0.05))
    # ground_truth_displacement = np.reshape(ground_truth_displacement, (-1,2))
    # y_predicted = q22.T
    y_predicted_resultant = ((y_predicted[:, 0]) ** 2 + (y_predicted[:, 1]) ** 2) ** 0.5
    ground_truth_displacement_resultant = (
        ((ground_truth_displacement[:, 0]) ** 2)
        + (ground_truth_displacement[:, 1]) ** 2
    ) ** 0.5

    # y_predicted_resultant = np.delete(y_predicted_resultant, np.where(np.logical_or(ground_truth_displacement_resultant > 0.05**2, ground_truth_displacement_resultant < -0.05**2)))
    # ground_truth_displacement_resultant = np.delete(ground_truth_displacement_resultant, np.where(np.logical_or(ground_truth_displacement_resultant > 0.05**2, ground_truth_displacement_resultant < -0.05**2)))

    # print(q11.shape, ground_truth_displacement.shape)
    # print(l1_loss(q11, ground_truth_displacement))
    return l1_loss(y_predicted_resultant, ground_truth_displacement_resultant)
    # x = np.abs(y_predicted_resultant - ground_truth_displacement_resultant)
    # print(x)
    # plt.imshow(1 - geometry[0].T / 255, cmap="Greys")
    # plt.colorbar()
    # plt.scatter(cords[:, 0], cords[:, 1], c=x, cmap="autumn", s=0.5)
    # plt.colorbar()
    # # plt.xlim(0,256)
    # # plt.ylim(0,256)
    # plt.show()
