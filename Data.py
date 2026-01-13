from PIL import Image
class YoloCustomDataset(Dataset):
    def __init__(self, images_dir, labels_dir, S, B, C):
        self.image_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
        self.labels_dir = labels_dir
        self.S = S
        self.B = B
        self.C = C
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(IMAGE_SIZE)
        ])

        # Generate class index from dataset if needed
        self.classes = {}
        index = 0
        for img_path in self.image_paths:
            txt_path = os.path.join(labels_dir, os.path.basename(img_path).replace(".jpg", ".txt")) # it just takes jpg and txt pairs since they have the same name
            if not os.path.exists(txt_path):
                continue
            # it checks if the object is in one of the classes
            with open(txt_path, "r") as f: 
                lines = f.readlines()
                for line in lines:
                    class_id = int(line.split()[0])
                    if class_id not in self.classes:
                        self.classes[class_id] = class_id  # here class_id is already numeric
                        index += 1

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        img_path = self.image_paths[i]
        txt_path = os.path.join(self.labels_dir, os.path.basename(img_path).replace(".jpg", ".txt"))

        # Load image
        img = Image.open(img_path).convert("RGB")
        # Apply transforms (resize + tensor)
        data = self.transform(img)
        # Keep a copy for visualization (same size as data)
        original_data = data.clone()

        # Grid size
        grid_size_x = data.size(2) / S
        grid_size_y = data.size(1) / S

        # Ground truth tensor SxSx(5*B + C)
        depth = 5 * B + C
        ground_truth = torch.zeros((S, S, depth))
        boxes = {}
        class_names = {}

        # Load labels from txt
        # It takes x_center, y_center, width, height all relative to the image size of the ground truth bounding boxes 
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                tokens = line.strip().split()
                class_id = int(tokens[0])
                x_center, y_center, width, height = map(float, tokens[1:]) 
                # (in the txt file data are written in the following way: class_id, x_center, y_center, width, height)

                # Just take the labels unstandardized 
                mid_x = x_center * IMAGE_SIZE[0]
                mid_y = y_center * IMAGE_SIZE[1]
                w = width * IMAGE_SIZE[0]
                h = height * IMAGE_SIZE[1]
                
                # We find the cell with inside the center  of the object
                col = int(mid_x // grid_size_x)
                row = int(mid_y // grid_size_y)

                if 0 <= col < S and 0 <= row < S:
                    cell = (row, col)
                    if cell not in class_names or class_id == class_names[cell]:
                        # One-hot class encoding
                        one_hot = torch.zeros(C)
                        one_hot[class_id] = 1.0
                        ground_truth[row, col, :C] = one_hot
                        class_names[cell] = class_id

                        # Bounding box
                        bbox_index = boxes.get(cell, 0)
                        if bbox_index < B:
                            bbox_truth = (
                                (mid_x - col * grid_size_x) / IMAGE_SIZE[0],  # X relative to cell
                                (mid_y - row * grid_size_y) / IMAGE_SIZE[1],  # Y relative to cell
                                w / IMAGE_SIZE[0],
                                h / IMAGE_SIZE[1],
                                1.0
                            )
                            bbox_start = 5 * bbox_index + C
                            ground_truth[row, col, bbox_start:] = torch.tensor(bbox_truth).repeat(B - bbox_index)
                            boxes[cell] = bbox_index + 1

        return data, ground_truth, original_data

# Load the dataset
full_dataset = YoloCustomDataset(images_dir = TRAIN_IMAGES_PATH, labels_dir= TRAIN_LABELS_PATH, S = S, B= B , C = C)
n_total = len(full_dataset)
n_test   = int(0.2 * n_total)
n_train = n_total - n_test 

train_set, test_set = random_split( 
    full_dataset,
    [n_train, n_test], 
    generator=torch.Generator().manual_seed(42)
)

print(f"Totale immagini: {n_total} | train: {n_train} | test: {n_test}") 

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    num_workers=0,
    persistent_workers=False,
    drop_last=True,
    shuffle=True
)
test_loader = DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    num_workers=0,
    persistent_workers=False,
    drop_last=True
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))