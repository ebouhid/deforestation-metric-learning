import timm
import torch
from dataset import SegmentClassificationDataset
from torch.utils.data import DataLoader
import argparse
import json
import os
import numpy as np
from tqdm import tqdm
from mahotas.features import haralick
from pytorch_lightning import LightningModule
from classification_models import ClassificationModel


class TrunkModelDML(torch.nn.Module):
    def __init__(self, backbone_name="resnet50", pretrained=True):
        super(TrunkModelDML, self).__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)

    def forward(self, x):
        return self.backbone(x)


class EmbedderModelDML(torch.nn.Module):
    def __init__(self, input_size, embedding_size):
        super(EmbedderModelDML, self).__init__()
        self.embedding_layer = torch.nn.Linear(input_size, embedding_size)

    def forward(self, x):
        return self.embedding_layer(x)


class EmbeddingModel(torch.nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True):
        super(EmbeddingModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, features_only=False)
        # Strip output layer
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.model_name = model_name
        self.pretrained = pretrained
    
    def forward(self, x):
        return self.model(x)

    def get_output_dims(self):
        some_tensor = torch.rand(1, 3, 28, 28)
        some_output = self.model(some_tensor)

        out_shapes = [x.shape for x in some_output]
        return out_shapes

class FineTunedEmbeddingModel(LightningModule):
    def __init__(self, ckpt_path: str):
        super(FineTunedEmbeddingModel, self).__init__()
        self.model = ClassificationModel.load_from_checkpoint(ckpt_path)
        self.model = torch.nn.Sequential(*list(self.model.model.children())[:-1])
        self.model = self.model.eval().to('cpu')
    
    def forward(self, x):
        return self.model(x)
    
    def get_output_dims(self):
        some_tensor = torch.rand(1, 3, 28, 28)
        some_output = self.model(some_tensor)

        out_shapes = [x.shape for x in some_output]
        return out_shapes
        

class HaralickGenerator(torch.nn.Module):
    def __init__(self):
        super(HaralickGenerator, self).__init__()

    def forward(self, x):
        batch_out = []
        for batch_item in range(x.shape[0]):
            out = []
            img = x[batch_item].permute(1, 2, 0).numpy()
            img *= 255
            img = img.astype(np.uint8)
            for ch in range(img.shape[2]):
                har_features = haralick(img[:, :, ch])
                out.append(har_features)
            batch_out.append(torch.tensor(np.array(out)))
        
        return torch.stack(batch_out)

    
    def get_output_dims(self):
        some_tensor = torch.rand(16, 3, 28, 28) * 255
        some_tensor = some_tensor.type(torch.uint8)
        some_output = self(some_tensor)

        out_shapes = [x.shape for x in some_output]
        return out_shapes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, choices=["resnet18", "resnet50", "resnet101", "haralick"])
    parser.add_argument("--pretrained", type=bool, default=True)
    parser.add_argument("--fine_tune_ckpt", type=str, default=None)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)

    args = parser.parse_args()

    TRAIN_REGIONS = ['x02', 'x06', 'x07', 'x09', 'x10']
    VAL_REGIONS = ['x03', 'x08']
    TEST_REGIONS = ['x01', 'x04']

    if args.model_name != "haralick":
        if args.fine_tune_ckpt is None:
            model = EmbeddingModel(args.model_name, args.pretrained)
        else:
            embedding_size = timm.create_model(args.model_name, pretrained=True).fc.in_features
            trunk = TrunkModelDML(backbone_name=args.model_name, pretrained=True)
            embedder = EmbedderModelDML(input_size=trunk.backbone.num_features, embedding_size=embedding_size)
            model = torch.nn.Sequential(trunk, embedder)
            model.load_state_dict(torch.load(args.fine_tune_ckpt, weights_only=True))
            model.get_output_dims = lambda : [x.shape for x in model(torch.rand(1, 3, 28, 28))]
            model.eval()
    elif args.model_name == "haralick":
        model = HaralickGenerator()
    

    train_ds = SegmentClassificationDataset(root_dir=args.input_dir, regions=TRAIN_REGIONS, transform=True)
    val_ds = SegmentClassificationDataset(root_dir=args.input_dir, regions=VAL_REGIONS, transform=False)
    test_ds = SegmentClassificationDataset(root_dir=args.input_dir, regions=TEST_REGIONS, transform=False)

    train_dataloader = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=23)
    val_dataloader = DataLoader(val_ds, batch_size=32, num_workers=23)
    test_dataloader = DataLoader(test_ds, batch_size=32, num_workers=23)

    metadata = dict()
    metadata["model_name"] = args.model_name
    metadata["pretrained"] = args.pretrained
    metadata["fine_tune_ckpt"] = args.fine_tune_ckpt
    metadata["train_regions"] = TRAIN_REGIONS
    metadata["val_regions"] = VAL_REGIONS
    metadata["test_regions"] = TEST_REGIONS
    metadata["output_dims"] = model.get_output_dims()

    os.makedirs(os.path.join(args.output_dir), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "train", "forest"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "train", "recent_def"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "val", "forest"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "val", "recent_def"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "test", "forest"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "test", "recent_def"), exist_ok=True)

    for split, dataloader in [("train", train_dataloader), ("val", val_dataloader), ("test", test_dataloader)]: 
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Processing {split}"):
            image = batch["image"]
            gt = batch["label"]
            embeddings = model(image)
            num_embs = embeddings.shape[0]
            for i in range(num_embs):
                emb = embeddings[i].detach().cpu().numpy()
                label = gt[i]
                if label == 0:
                    label_name = "forest"
                else:
                    label_name = "recent_def"
                emb_path = os.path.join(args.output_dir, split, label_name, f"{idx}_{i}.npy")
                np.save(emb_path, emb)
    
    metadata["train_size"] = len(train_ds)
    metadata["train_forest"] = len([x for x in train_ds.labels if x == 0])
    metadata["train_recent_def"] = len([x for x in train_ds.labels if x == 1])

    metadata["val_size"] = len(val_ds)
    metadata["val_forest"] = len([x for x in val_ds.labels if x == 0])
    metadata["val_recent_def"] = len([x for x in val_ds.labels if x == 1])

    metadata["test_size"] = len(test_ds)
    metadata["test_forest"] = len([x for x in test_ds.labels if x == 0])
    metadata["test_recent_def"] = len([x for x in test_ds.labels if x == 1])

    with open(f"{args.output_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
