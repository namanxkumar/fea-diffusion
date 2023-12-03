from datagen.generate import generate_data

generate_data(
    data_dir="test",
    image_size=256,
    num_plates=1,
    start_plate=None, # None for starting from scratch, or a number to continue from
    conditions_per_plate=1,
    mesh_size=1e-2,
    save_displacement=True,
    save_strain=False,
    save_stress=False,
    save_meshes=True
)