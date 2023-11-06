from datagen.generate import generate_data

generate_data(
    data_dir="data",
    image_size=512,
    num_plates=500,
    start_plate=12,
    conditions_per_plate=4,
    mesh_size=1e-2,
    save_displacement=True,
    save_strain=False,
    save_stress=False
)