from pycocotools.coco import COCO
import numpy as np
import os
from PIL import Image

coco_json_path = r'C:\Users\HP\MoonNav2\dataset\images\val\_annotations.coco.json'
images_dir = r'C:\Users\HP\MoonNav2\dataset\images\val'
masks_dir = r'C:\Users\HP\MoonNav2\dataset\masks\val'

os.makedirs(masks_dir, exist_ok=True)

coco = COCO(coco_json_path)

for img_id in coco.getImgIds():
    img_info = coco.loadImgs(img_id)[0]
    img_file_name = img_info['file_name']
    height, width = img_info['height'], img_info['width']

    
    mask = np.zeros((height, width), dtype=np.uint16)  

   
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    for idx, ann in enumerate(anns, start=1):
        
        if 'segmentation' in ann and ann['segmentation']:
            try:
                ann_mask = coco.annToMask(ann)
                mask[ann_mask == 1] = idx 
            except Exception as e:
                print(f"Error processing annotation {ann['id']} in {img_file_name}: {e}")
        else:
            print(f"Skipping annotation {ann.get('id', 'unknown')} in {img_file_name}: empty segmentation.")

    mask_path = os.path.join(masks_dir, img_file_name.replace('.jpg', '_mask.png'))
    print(f"Saving mask to: {mask_path}")
    try:
        Image.fromarray(mask).save(mask_path)
        print(f"Saved mask for {img_file_name} at {mask_path}")
    except Exception as e:
        print(f"Failed to save mask for {img_file_name}: {e}")
