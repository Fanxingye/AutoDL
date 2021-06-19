import os
import shutil
for data_type in ["train","val","test"]:
    data_dir = f"/data/dataset/UKCarsDataset/split/{data_type}"
    # os.remove(data_dir+"/images.txt")
    data_labels = os.listdir(data_dir)
    with open("images.txt", "w")as f:
        for index, data_label in enumerate(data_labels):
            imgs = os.listdir(os.path.join(data_dir, data_label))
            for img in imgs:
                path = os.path.join(data_label, img)
                path = path.replace("\\", "/")
                f.write(f"{path} {index}\n")
                shutil.copy("images.txt", f"/data/dataset/UKCarsDataset/{data_type}.txt")
