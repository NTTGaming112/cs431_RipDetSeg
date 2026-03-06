import os
import shutil
import imagehash
from PIL import Image
from tqdm import tqdm
import pandas as pd
import random

dataset_path = r"cs431_RipDetSeg\RipDetSeg_v1.1.6_train\RipDetSeg_v1.1.6_train\train_images"
label_source_path = r"cs431_RipDetSeg\RipDetSeg_v1.1.6_train\RipDetSeg_v1.1.6_train\train_labels_segmentation" 
output_path = r"cs431_RipDetSeg\Clustered_Images"
final_dataset_path = r"cs431_RipDetSeg\RipCurrent_Final_Dataset"

threshold = 10 
clusters = {} 
train_ratio = 0.8

#phân nhóm
def cluster_with_labels(dataset_path, label_source_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img_name in tqdm(files):
        img_path = os.path.join(dataset_path, img_name)
        try:
            with Image.open(img_path) as img:
                h = imagehash.phash(img)
            
            found_cluster = False
            for ref_hash in clusters.keys():    
                if h - ref_hash < threshold:
                    clusters[ref_hash].append(img_name)
                    found_cluster = True
                    break
            if not found_cluster:
                clusters[h] = [img_name]
        except: continue

    total_with_label = 0
    total_no_label = 0    

    all_label_files = os.listdir(label_source_path)
    label_map = {os.path.splitext(f)[0]: f for f in all_label_files}

    for i, (h_goc, member_files) in enumerate(tqdm(clusters.items())):
        group_name = f"Group_{i+1:03d}_count_{len(member_files)}"
        group_dir = os.path.join(output_path, group_name)
        
        img_out_dir = os.path.join(group_dir, "images")
        lbl_out_dir = os.path.join(group_dir, "labels")
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(lbl_out_dir, exist_ok=True)
        
        for f_name in member_files:
            shutil.copy2(os.path.join(dataset_path, f_name), os.path.join(img_out_dir, f_name))
            
            base_name = os.path.splitext(f_name)[0]
            
            if base_name in label_map:
                actual_label_file = label_map[base_name]
                shutil.copy2(os.path.join(label_source_path, actual_label_file), 
                             os.path.join(lbl_out_dir, actual_label_file))
                total_with_label += 1
            else:
                total_no_label += 1

    print(f"images with labels: {total_with_label}")
    print(f"images no labels: {total_no_label}")

#resize 1024 hoặc.....
def resize_dataset(target_size=1024):
    groups = [d for d in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, d))]
    for g in tqdm(groups):
        img_folder = os.path.join(output_path, g, "images")
        for f in os.listdir(img_folder):
            f_path = os.path.join(img_folder, f)
            with Image.open(f_path) as img:
                img = img.convert("RGB")
                from PIL import ImageOps
                img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
                new_img = ImageOps.pad(img, (target_size, target_size), color=(114, 114, 114))
                new_img.save(f_path)

#chia độ khó từng nhóm
def cluster_group(output_path):
    analysis_data = []
    groups = [d for d in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, d))]
    
    for g in groups:
        g_path = os.path.join(output_path, g)
        n_img = len(os.listdir(os.path.join(g_path, "images")))
        n_lbl = len(os.listdir(os.path.join(g_path, "labels")))
        ratio = n_lbl / n_img if n_img > 0 else 0
        
        if ratio == 1.0: diff = "Easy"
        elif ratio == 0.0: diff = "Hard (Pure Background)"
        else: diff = "Medium (Mixed)"
            
        analysis_data.append({"Group": g, "Ratio": ratio, "Difficulty": diff})
    
    df = pd.DataFrame(analysis_data)
    print(df['Difficulty'].value_counts())
    df.to_csv(os.path.join(output_path, "difficulty_summary.csv"), index=False)

#chia dataset cuối cùng
def build_dataset():
    csv_summary_path = os.path.join(output_path, "difficulty_summary.csv")
    df = pd.read_csv(csv_summary_path)
    
    for split in ['train', 'val']:
        for folder in ['images', 'labels']:
            os.makedirs(os.path.join(final_dataset_path, split, folder), exist_ok=True)

    train_groups = []
    val_groups = []

    for diff_type in df['Difficulty'].unique():
        sub_df = df[df['Difficulty'] == diff_type]
        group_list = sub_df['Group'].tolist()
        
        random.seed(42)
        random.shuffle(group_list)
        
        split_idx = int(len(group_list) * train_ratio)
        train_groups.extend(group_list[:split_idx])
        val_groups.extend(group_list[split_idx:])
        
        print(f"[{diff_type}]: Train {split_idx} groups, Val {len(group_list)-split_idx} groups")

    def copy_files(group_names, split_name):
        for g_name in tqdm(group_names):
            g_path = os.path.join(output_path, g_name)

            img_src = os.path.join(g_path, "images")
            for f in os.listdir(img_src):
                shutil.copy2(os.path.join(img_src, f), os.path.join(final_dataset_path, split_name, 'images', f))
            
            lbl_src = os.path.join(g_path, "labels")
            for f in os.listdir(lbl_src):
                shutil.copy2(os.path.join(lbl_src, f), os.path.join(final_dataset_path, split_name, 'labels', f))

    copy_files(train_groups, 'train')
    copy_files(val_groups, 'val')


cluster_with_labels(dataset_path, label_source_path, output_path)
resize_dataset(1024)
cluster_group(output_path)
build_dataset()