import torch.utils.data
import pycocotools.coco
import os


class CocoObjectDataset(torch.utils.data.Dataset):
    
    def __init__(self, images_dir, annotation_file, category_name, transforms=None):
        coco = pycocotools.coco.COCO(annotation_file)
        cat_id = coco.getCatIds(category_name)[0]
        self.cat = coco.cats[cat_id]
        img_ids = coco.getImgIds(catIds=cat_id)
        self.entries = []
        for img_id in img_ids:
            img = coco.imgs[img_id]
            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_id)
            anns = [coco.anns[ann_id] for ann_id in ann_ids]
            self.entries.append({
                'path': os.path.join(images_dir, img['file_name']),
                'img': img,
                'anns': anns
            })
        self.transforms = transforms
    
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        entry = self.entries[idx]
        
        if self.transforms is not None:
            return self.transforms(entry)
        
        return entry