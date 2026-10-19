"""Entrenamiento reproducible. Requiere manifest.csv real; no crea datos ni métricas ficticias."""
import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from transformers import CLIPModel, CLIPProcessor

CLASSES = ['Clear protein', 'Creatina', 'Preentreno']
PROMPTS = ['a container of clear protein supplement powder', 'a container of creatine supplement powder', 'a container of pre workout sports supplement powder']
SEED = 42


def prepare(root='.'):
    root = Path(root).resolve()
    manifest = root / 'datos/manifest.csv'
    df = pd.read_csv(manifest).fillna('')
    required = ['path', 'class', 'group_id', 'source_url', 'license']
    if not all(c in df for c in required) or df.empty:
        raise ValueError('Faltan imágenes reales. Completa datos/manifest.csv; no se puede entrenar con una plantilla vacía.')
    if (df[required].astype(str).apply(lambda s: s.str.strip()) == '').any().any():
        raise ValueError('Cada imagen requiere ruta, clase, grupo, fuente y licencia o permiso.')
    if set(df['class']) != set(CLASSES):
        raise ValueError('Las clases deben coincidir exactamente con CLASSES (tres categorías).')
    if (df.groupby('group_id')['class'].nunique() > 1).any():
        raise ValueError('Un grupo no puede tener etiquetas contradictorias.')
    resolved = []
    hashes = []
    for name in df.path:
        path = (root / str(name)).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Las imágenes deben estar dentro del proyecto.')
        with Image.open(path) as im:
            im.verify()
        resolved.append(str(path))
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    df['sha256'] = hashes
    if df.sha256.duplicated().any():
        raise ValueError('Hay archivos duplicados exactos. Elimina duplicados antes de dividir.')
    df['_absolute'] = resolved
    if 'split' not in df or not set(df['split']).issubset({'train','val','test'}):
        raise ValueError('Ejecuta prepare_coco: se deben conservar las particiones de la exportación COCO.')
    if (df.groupby('group_id')['split'].nunique() > 1).any():
        raise ValueError('Un grupo cruza particiones; revisar antes de entrenar.')
    if not all(set(df.loc[df.split == split, 'class']) == set(CLASSES) for split in ['train', 'val', 'test']):
        raise ValueError('Cada partición debe incluir las tres clases.')
    (root / 'resultados').mkdir(exist_ok=True)
    df.drop(columns=['_absolute']).to_csv(root / 'resultados/division.csv', index=False)
    counts = pd.crosstab(df['class'], df.split).reindex(CLASSES)
    counts.to_csv(root / 'resultados/recuento.csv')
    print(counts)
    print('Se conservan train/valid/test de Roboflow; no se redistribuyen las imágenes aumentadas.')
    return df


class Images(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.transform = ResNet18_Weights.DEFAULT.transforms()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with Image.open(row['_absolute']) as im:
            im = ImageOps.exif_transpose(im).convert('RGB')
            image = self.transform(im)
        return image, CLASSES.index(row['class'])


def measure(y, scores):
    pred = np.argmax(scores, axis=1)
    return {'accuracy': float(accuracy_score(y, pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
            'report': classification_report(y, pred, labels=list(range(len(CLASSES))), target_names=CLASSES, zero_division=0, output_dict=True),
            'confusion_matrix': confusion_matrix(y, pred, labels=list(range(len(CLASSES)))).tolist()}


def evaluate_resnet(model, data, device):
    model.eval()
    labels, scores = [], []
    with torch.inference_mode():
        for x, y in DataLoader(Images(data), batch_size=16, shuffle=False):
            scores.extend(model(x.to(device)).softmax(-1).cpu().tolist())
            labels.extend(y.tolist())
    return measure(labels, scores), scores


def evaluate_clip(data, device):
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device).eval()
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    labels, scores = [], []
    with torch.inference_mode():
        for _, row in data.iterrows():
            with Image.open(row['_absolute']) as im:
                inputs = processor(text=PROMPTS, images=ImageOps.exif_transpose(im).convert('RGB'), return_tensors='pt', padding=True).to(device)
            scores.append(model(**inputs).logits_per_image.softmax(-1)[0].cpu().tolist())
            labels.append(CLASSES.index(row['class']))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return measure(labels, scores), scores


def run(root='.', epochs=12):
    root = Path(root).resolve()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    df = prepare(root)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train, val, test = [df[df.split == s] for s in ['train', 'val', 'test']]
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.to(device)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
    loader = DataLoader(Images(train), batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    best, history, best_state = -1, [], None
    for epoch in range(epochs):
        # Mantener estadísticas BatchNorm del backbone preentrenado.
        model.eval()
        model.fc.train()
        total = 0
        for x, y in loader:
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
        metrics, _ = evaluate_resnet(model, val, device)
        history.append({'epoch': epoch + 1, 'train_loss': total / len(train), 'val_balanced_accuracy': metrics['balanced_accuracy']})
        print(history[-1])
        if metrics['balanced_accuracy'] > best:
            best = metrics['balanced_accuracy']
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
    model.load_state_dict(best_state)
    val_resnet, _ = evaluate_resnet(model, val, device)
    val_clip, _ = evaluate_clip(val, device)
    selected = 'resnet18' if val_resnet['balanced_accuracy'] >= val_clip['balanced_accuracy'] else 'clip'
    # El ganador se fija ANTES de mirar el test. No ajustar después con sus resultados.
    test_resnet, resnet_scores = evaluate_resnet(model, test, device)
    test_clip, clip_scores = evaluate_clip(test, device)
    all_metrics = {'validation': {'resnet18': val_resnet, 'clip': val_clip},
                   'test': {'resnet18': test_resnet, 'clip': test_clip}, 'selected': selected,
                   'selection_rule': 'Mayor balanced accuracy en validación; empate favorece ResNet18',
                   'seed': SEED, 'test_size': len(test)}
    out = root / 'modelo'
    out.mkdir(exist_ok=True)
    torch.save(best_state, out / 'resnet18.pt')
    (out / 'classes.json').write_text(json.dumps(CLASSES, ensure_ascii=False, indent=2))
    (out / 'deployment.json').write_text(json.dumps({'selected': selected, 'classes': CLASSES, 'prompts': PROMPTS,
        'test': all_metrics['test'][selected], 'test_size': len(test), 'clip_id': 'openai/clip-vit-base-patch32',
        'resnet_preprocessing': 'ResNet18_Weights.IMAGENET1K_V1.transforms()'}, ensure_ascii=False, indent=2))
    (root / 'resultados/metrics.json').write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2))
    pd.DataFrame(history).to_csv(root / 'resultados/history.csv', index=False)
    predictions = test[['path', 'class', 'group_id']].copy()
    for name, values in [('resnet18', resnet_scores), ('clip', clip_scores)]:
        predictions[name + '_pred'] = [CLASSES[int(np.argmax(v))] for v in values]
        predictions[name + '_confidence'] = [max(v) for v in values]
    predictions.to_csv(root / 'resultados/predicciones_test.csv', index=False)
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    for name, metrics in all_metrics['test'].items():
        fig, ax = plt.subplots(figsize=(9, 7))
        ConfusionMatrixDisplay(np.array(metrics['confusion_matrix']), display_labels=CLASSES).plot(ax=ax, xticks_rotation=35, colorbar=False)
        ax.set_title(name + ' - conjunto de prueba')
        fig.tight_layout()
        fig.savefig(root / ('resultados/confusion_' + name + '.png'), dpi=160)
        plt.close(fig)
    print(json.dumps(all_metrics, ensure_ascii=False, indent=2))
    return all_metrics


if __name__ == '__main__':
    run()
