import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms
import segmentation_models_pytorch as smp

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
RESIZE_SIZE = (256, 256) 

class LunarDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(IMG_EXTS)])
        self.masks = sorted([f for f in os.listdir(masks_dir) if f.lower().endswith('.png')])
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.resize = transforms.Resize(RESIZE_SIZE)
        self.pairs = [(img, msk) for img, msk in zip(self.images, self.masks)
                     if os.path.splitext(img)[0] == os.path.splitext(msk)[0].replace('_mask', '')]
        if len(self.pairs) == 0:
            raise RuntimeError('No matching image-mask pairs found! Check your filenames.')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_name, mask_name = self.pairs[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)
        img = read_image(img_path).float() / 255.0
        img = self.resize(img)
        mask = read_image(mask_path).float() / 255.0  
        mask = mask.squeeze(0) 
        mask = self.resize(mask.unsqueeze(0)).squeeze(0) 
        mask = (mask > 0.5).float()
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        return img, mask


def main():
    train_images_dir = r"C:/Users/HP/MoonNav2/dataset/images/train"
    train_masks_dir = r"C:/Users/HP/MoonNav2/dataset/masks/train"
    val_images_dir = r"C:/Users/HP/MoonNav2/dataset/images/val"
    val_masks_dir = r"C:/Users/HP/MoonNav2/dataset/masks/val"

    train_dataset = LunarDataset(train_images_dir, train_masks_dir)
    val_dataset = LunarDataset(val_images_dir, val_masks_dir)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

    model = smp.DeepLabV3Plus(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            preds = preds.squeeze(1)
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                preds = preds.squeeze(1)
                loss = criterion(preds, masks)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}")

    model_path = "deeplabv3_lunar.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
