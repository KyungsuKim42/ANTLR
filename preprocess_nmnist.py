import numpy as np
from pathlib import Path
import os
import pdb



for split in ["Test"]:
    data_path = Path(f"./dataset/N-MNIST/{split}")

    # Make label.csv
    if split=="Train":
        label = -1 * np.ones(60000)
    else:
        label = -1 * np.ones(10000)
    for target_class in range(10):
        f_list = list((data_path / str(target_class)).iterdir())
        for f in f_list:
            index = int(str(f).split('/')[-1].split('.')[0])
            label[index-1] = target_class
    assert (label>=0).all()
    np.savetxt(data_path / "label.csv" , label, delimiter=",", fmt='%d')

    # Move all files to the parent directory.
    for target_class in range(10):
        print(f"Moving files from {split}/{target_class}")
        os.system(f"mv ./dataset/N-MNIST/{split}/{target_class}/* ./dataset/N-MNIST/{split}/")
        os.system(f"rm -r ./dataset/N-MNIST/{split}/{target_class}")


    import pdb; pdb.set_trace()
