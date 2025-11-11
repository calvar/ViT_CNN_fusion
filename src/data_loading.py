import os
import numpy as np
from pathlib import Path
import csv
from PIL import Image
import kagglehub

import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit


def download_datasets():
    ############ Chest X-ray Pneumonia dataset ############ 
    chest_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    nih_path = kagglehub.dataset_download("nih-chest-xrays/data")

    return chest_path, nih_path


def get_paths(chest_path, nih_path):
    KAGGLE_ROOT = Path(chest_path + "/chest_xray")
    if os.path.exists(KAGGLE_ROOT):
        print("Directory exists:", KAGGLE_ROOT)
        print("Contents:", os.listdir(KAGGLE_ROOT))
    else:
        print("Directory does not exist:", KAGGLE_ROOT)
    
    NIH_ROOT = Path(nih_path)
    if os.path.exists(NIH_ROOT):
        print("Directory exists:", NIH_ROOT)
        print("Contents:", os.listdir(NIH_ROOT))
    else:
        print("Directory does not exist:", NIH_ROOT)
    
    # -----------------
    # Paths
    # -----------------
    K_TRAIN = KAGGLE_ROOT / "train"
    K_VAL   = KAGGLE_ROOT / "val"
    K_TEST  = KAGGLE_ROOT / "test"

    NIH_LABELS = NIH_ROOT / "Data_Entry_2017.csv"
    NIH_TVLIST = NIH_ROOT / "train_val_list.txt"
    NIH_TLIST  = NIH_ROOT / "test_list.txt"
    NIH_IMAGE_DIRS = [NIH_ROOT / f"images_{i:03d}" for i in range(1, 13)]
    return K_TRAIN, K_VAL, K_TEST, NIH_LABELS, NIH_TVLIST, NIH_TLIST, NIH_IMAGE_DIRS


# -----------------
# NIH helpers: index all filenames to absolute paths
# -----------------
# --- NEW: robust recursive indexer for NIH images (handles images_0xx/images/*) ---
def index_nih_images(img_dirs):
    """
    Build {filename -> absolute Path} for all NIH images.
    Handles layouts like:
      images_001/images/*.png (common Kaggle mirror)
      images_001/*.png        (some mirrors)
    """
    exts = {".png", ".jpg", ".jpeg"}
    index = {}
    for d in img_dirs:
        if not d.exists():
            continue
        # Preferred nested folder
        roots = []
        if (d / "images").exists():
            roots.append(d / "images")
        else:
            roots.append(d)

        for root in roots:
            # Use rglob in case there are further subfolders
            for p in root.rglob("*"):
                if p.is_file() and p.suffix.lower() in exts:
                    index[p.name] = p.resolve()

    # quick diag
    print(f"[NIH] Indexed {len(index):,} image files from {len(img_dirs)} dirs.")
    return index

def read_nih_labels_csv(csv_path):
    """Return dict: filename -> set(labels). NIH labels are pipe-separated."""
    label_map = {}
    with open(csv_path, newline='') as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            fname = row["Image Index"].strip()
            labs  = [s.strip() for s in row["Finding Labels"].split("|")]
            label_map[fname] = set(labs)
    return label_map

def read_list_txt(txt_path):
    """Lines are filenames, possibly with paths; take the last path component."""
    out = []
    with open(txt_path, "r") as f:
        for line in f:
            name = line.strip().split("/")[-1]
            if name:
                out.append(name)
    return out



# -----------------
# Build NIH splits (binary: 0=Normal, 1=Pneumonia)
# -----------------
class NIHBinaryDataset(Dataset):
    def __init__(self, items, transform=None):
        """
        items: list of (path, label_int)
        """
        self.items = items
        self.transform = transform
        self.targets = [y for _, y in items]   # for samplers/metrics

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, y

def make_nih_binary_splits(
    nih_root_dirs,
    labels_csv,
    tv_list_txt=None,
    test_list_txt=None,
    val_ratio=0.10,
    test_ratio=0.10,
    seed=42):

    rng = np.random.RandomState(seed)
    fname_to_path   = index_nih_images(nih_root_dirs)
    fname_to_labels = read_nih_labels_csv(labels_csv)

    # binary label fn
    def label_fn(fname):
        labs = fname_to_labels.get(fname, set())
        if "Pneumonia" in labs:
            return 1
        # keep strictly-normal only
        if labs == {"No Finding"} or ("No Finding" in labs and len(labs) == 1):
            return 0
        return None

    # --- Build the full eligible pool (No Finding / Pneumonia only) ---
    pool_items = []
    for fname, p in fname_to_path.items():
        y = label_fn(fname)
        if y is not None:
            pool_items.append((fname, p, y))

    if len(pool_items) == 0:
        raise RuntimeError("NIH: No eligible samples found. Check NIH_ROOT / CSV columns.")

    # Helper to convert a list file to a filtered (fname,y) list
    def list_to_items(list_path):
        names = read_list_txt(list_path)
        keep = []
        for n in names:
            if n in fname_to_path:
                y = label_fn(n)
                if y is not None:
                    keep.append((n, fname_to_path[n], y))
        return keep

    use_official = False
    tv_items = []
    test_items = []

    # Try official lists if provided
    if tv_list_txt is not None and test_list_txt is not None and \
       Path(tv_list_txt).exists() and Path(test_list_txt).exists():
        tv_items   = list_to_items(tv_list_txt)
        test_items = list_to_items(test_list_txt)
        if len(tv_items) > 0 and len(test_items) > 0:
            use_official = True

    if use_official:
        # stratified split tv -> train/val
        y_tv = np.array([y for _, _, y in tv_items])
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        tr_idx, va_idx = next(sss.split(np.zeros(len(y_tv)), y_tv))
        train_items = [tv_items[i] for i in tr_idx]
        val_items   = [tv_items[i] for i in va_idx]
        print(f"NIH (official lists): tv={len(tv_items)} test={len(test_items)} "
              f"→ train={len(train_items)} val={len(val_items)}")
    else:
        # Fallback: stratified split on the full pool into train/val/test
        fnames  = np.array([f for f, _, _ in pool_items])
        paths   = np.array([p for _, p, _ in pool_items])
        labels  = np.array([y for _, _, y in pool_items])

        # First split out test
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
        trainval_idx, test_idx = next(sss1.split(np.zeros(len(labels)), labels))
        fn_tv, fn_te = fnames[trainval_idx], fnames[test_idx]
        p_tv,  p_te  = paths[trainval_idx],  paths[test_idx]
        y_tv,  y_te  = labels[trainval_idx], labels[test_idx]

        # Then split train/val
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        tr_idx, va_idx = next(sss2.split(np.zeros(len(y_tv)), y_tv))

        train_items = list(zip(p_tv[tr_idx], y_tv[tr_idx]))
        val_items   = list(zip(p_tv[va_idx], y_tv[va_idx]))
        test_items  = list(zip(p_te,         y_te))

        print(f"NIH (fallback split): pool={len(pool_items)} "
              f"→ train={len(train_items)} val={len(val_items)} test={len(test_items)}")

    # Normalize tuples to (path,label) for the Dataset
    def norm(items):
        out = []
        for it in items:
            if isinstance(it[0], Path):
                # already (path,label)
                out.append((it[0], int(it[1])))
            else:
                # (fname, path, label)
                out.append((it[1], int(it[2])))
        return out

    train_items = norm(train_items)
    val_items   = norm(val_items)
    test_items  = norm(test_items)

    return train_items, val_items, test_items


def concat_targets(dataset: ConcatDataset):
    """Return list of labels for a ConcatDataset of ImageFolder + NIHBinaryDataset."""
    labels = []
    for ds in dataset.datasets:
        if isinstance(ds, datasets.ImageFolder):
            # ImageFolder: targets are the second element in samples
            labels.extend([y for _, y in ds.samples])
        elif hasattr(ds, "targets"):
            labels.extend(list(ds.targets))
        else:
            raise ValueError("Unknown dataset type inside ConcatDataset.")
    return np.array(labels, dtype=int)



def balance_classes(labels, nclasses, dataset):
    # compute per-class counts and balanced sample weights
    class_count = np.bincount(labels, minlength=nclasses)
    max_count = class_count.max()
    # Oversample minority class to match majority by per-sample weights
    class_weight = {c: (max_count / (class_count[c] if class_count[c] > 0 else 1))
                    for c in range(nclasses)}

    sample_weights = []
    offset = 0
    for ds in dataset.datasets:
        if isinstance(ds, datasets.ImageFolder):
            ys = [y for _, y in ds.samples]
        else:
            ys = ds.targets
        sample_weights.extend([class_weight[int(y)] for y in ys])
        offset += len(ys)
    
    sample_weights = torch.DoubleTensor(sample_weights)
    balanced_sampler = WeightedRandomSampler(
        weights=sample_weights,              # <-- use 'weights' not 'sample_weights'
        num_samples=len(sample_weights),     # one "epoch" ~ same # samples as dataset
        replacement=True)
    
    return balanced_sampler




# -----------------
# (Optional) CutMix collate — keep your class if you want it
# -----------------
class CutMixCollate:
    def __init__(self, alpha=1.0, p_cutmix=0.5, seed=42):
        self.alpha = alpha; self.p = p_cutmix; self.rng = np.random.RandomState(seed)
    @staticmethod
    def _rand_bbox(H, W, lam, rng):
        cut_ratio = np.sqrt(1. - lam)
        cut_w = int(W * cut_ratio); cut_h = int(H * cut_ratio)
        cx = rng.randint(W); cy = rng.randint(H)
        x1 = np.clip(cx - cut_w // 2, 0, W); y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W); y2 = np.clip(cy + cut_h // 2, 0, H)
        return x1, y1, x2, y2
    def __call__(self, batch):
        images = torch.stack([b[0] for b in batch], dim=0)
        targets = torch.tensor([b[1] for b in batch], dtype=torch.long)
        if self.rng.rand() > self.p or len(batch) < 2:
            return images, targets
        B, C, H, W = images.shape

        lam = np.random.beta(self.alpha, self.alpha)
        perm = torch.randperm(B)
        x1, y1, x2, y2 = self._rand_bbox(H, W, lam, self.rng)
        images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
        box_area = (x2 - x1) * (y2 - y1)
        lam_adj = 1.0 - box_area / float(H * W)
        return {"images": images, "targets": targets, "targets_perm": targets[perm], "lam": float(lam_adj), "is_cutmix": True}
    

# Image normalization transforms.
def normalize_transform(pretrained):
    if pretrained: # Normalization for pre-trained weights.
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
    else: # Normalization when training from scratch.
        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    return normalize

#---------------------------------------------------
# Load data into DataLoaders
#---------------------------------------------------

def load_data(img_size, batch_size, num_workers, pin_memory,
              use_cutmix=False, cutmix_alpha=1.0, cutmix_p=0.5,
              pretrained=True):
    # Download datasets if not already done
    chest_path, nih_path = download_datasets()
    print("Path to chest x-ray dataset files:", chest_path)
    print("Path to NIH chest x-ray dataset files:", nih_path)

    # Get paths
    K_TRAIN, K_VAL, K_TEST, NIH_LABELS, NIH_TVLIST, NIH_TLIST, NIH_IMAGE_DIRS = get_paths(chest_path, nih_path)

    # Transforms
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=7),
        transforms.ToTensor(),
        normalize_transform(pretrained)
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize_transform(pretrained)
    ])

    # Kaggle datasets (ImageFolder)
    k_train_ds = datasets.ImageFolder(str(K_TRAIN), transform=train_tfms)
    k_val_ds   = datasets.ImageFolder(str(K_VAL),   transform=eval_tfms)
    k_test_ds  = datasets.ImageFolder(str(K_TEST),  transform=eval_tfms)

    # Sanity: classes should be ['NORMAL', 'PNEUMONIA'] in that order
    print("Kaggle classes:", k_train_ds.classes)

    # NIH datasets (custom binary)
    nih_train_items, nih_val_items, nih_test_items = make_nih_binary_splits(
        NIH_IMAGE_DIRS, NIH_LABELS, NIH_TVLIST, NIH_TLIST, 
        val_ratio=0.10, test_ratio=0.10, seed=42
    )

    nih_train_ds = NIHBinaryDataset(nih_train_items, transform=train_tfms)
    nih_val_ds   = NIHBinaryDataset(nih_val_items,   transform=eval_tfms)
    nih_test_ds  = NIHBinaryDataset(nih_test_items,  transform=eval_tfms)

    print(f"NIH counts — train:{len(nih_train_ds)}  val:{len(nih_val_ds)}  test:{len(nih_test_ds)}")

    # Merge Kaggle + NIH per split
    train_ds = ConcatDataset([k_train_ds, nih_train_ds])
    val_ds   = ConcatDataset([k_val_ds,   nih_val_ds])
    test_ds  = ConcatDataset([k_test_ds,  nih_test_ds])

    train_targets = concat_targets(train_ds)
    balanced_sampler = balance_classes(train_targets, 2, train_ds)
    train_collate = CutMixCollate(alpha=cutmix_alpha, p_cutmix=cutmix_p, seed=42) if use_cutmix else None

    # DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=(balanced_sampler is None),
        num_workers=num_workers, pin_memory=pin_memory,
        sampler=balanced_sampler,
        collate_fn=train_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader, k_train_ds.classes