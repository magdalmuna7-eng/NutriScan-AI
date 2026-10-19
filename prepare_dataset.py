"""Convierte la exportación COCO de Beverage Detection v12 a clasificación.
Conserva las particiones del autor; no redistribuye aumentos de entrenamiento.
"""
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image

MAPPING = {'clear protein': 'Clear protein', 'creatine': 'Creatina', 'preworkout': 'Preentreno'}
SOURCE = 'https://universe.roboflow.com/tribe-tagging/beverage-detection-stpvk/dataset/12'


def prepare_coco(raw_dir, project='.'):
    raw_dir, project = Path(raw_dir).resolve(), Path(project).resolve()
    annotation_files = list(raw_dir.rglob('_annotations.coco.json'))
    if not annotation_files:
        raise ValueError('No se encontró _annotations.coco.json. Descarga la versión 12 como COCO.')
    rows, exclusions, seen, group_splits = [], [], {}, {}
    for annotation_file in sorted(annotation_files):
        split = {'train': 'train', 'valid': 'val', 'val': 'val', 'test': 'test'}.get(annotation_file.parent.name)
        if split is None:
            raise ValueError('Partición desconocida: ' + str(annotation_file.parent))
        data = json.loads(annotation_file.read_text())
        categories = {c['id']: c['name'].strip().lower() for c in data['categories']}
        annotations = defaultdict(list)
        for a in data['annotations']:
            annotations[a['image_id']].append(a)
        for item in data['images']:
            objects = annotations[item['id']]
            # Se usa la foto completa sólo si contiene una anotación de una clase objetivo.
            # Evita entrenar con un recorte y después evaluar fotos completas.
            if len(objects) != 1 or categories.get(objects[0]['category_id']) not in MAPPING:
                exclusions.append({'file': item['file_name'], 'split': split, 'reason': 'sin un único objeto objetivo anotado'})
                continue
            source_path = (annotation_file.parent / item['file_name']).resolve()
            if not source_path.is_relative_to(raw_dir):
                raise ValueError('Ruta de imagen fuera de la descarga')
            with Image.open(source_path) as im:
                im.verify()
            label = MAPPING[categories[objects[0]['category_id']]]
            digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
            if digest in seen:
                old_split, old_label = seen[digest]
                if old_split != split or old_label != label:
                    raise ValueError('Duplicado entre particiones o etiquetas contradictorias. Revisar antes de entrenar.')
                exclusions.append({'file': item['file_name'], 'split': split, 'reason': 'duplicado exacto dentro de partición'})
                continue
            seen[digest] = (split, label)
            group = Path(item['file_name']).name.split('.rf.')[0]
            if group in group_splits and group_splits[group] != split:
                raise ValueError('Un nombre de origen aparece en más de una partición: ' + group)
            group_splits[group] = split
            target = project / 'datos/imagenes' / split / (digest + source_path.suffix.lower())
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, target)
            rows.append({'path': str(target.relative_to(project)), 'class': label, 'group_id': group,
                         'source_url': SOURCE, 'license': 'CC BY 4.0 - tribe tagging; filtrado a tres clases',
                         'split': split, 'original_file': item['file_name']})
    if not rows:
        raise ValueError('No quedaron imágenes válidas de las tres clases seleccionadas.')
    frame = pd.DataFrame(rows)
    counts = pd.crosstab(frame['class'], frame['split']).reindex(index=list(MAPPING.values()), columns=['train','val','test'], fill_value=0).fillna(0).astype(int)
    if (counts == 0).any().any():
        raise ValueError('Falta una clase en alguna partición. No entrenar todavía. Recuento: ' + counts.to_json())
    (project / 'datos').mkdir(exist_ok=True)
    frame.to_csv(project / 'datos/manifest.csv', index=False)
    counts.to_csv(project / 'datos/recuento_filtrado.csv')
    pd.DataFrame(exclusions, columns=['file','split','reason']).to_csv(project / 'datos/exclusiones.csv', index=False)
    print('Imágenes después del filtrado (no son los totales del dataset original):')
    print(counts)
    print('Revisión pendiente: etiquetas, series de fotos similares, productos compartidos entre particiones y clases con pocas muestras.')
    return frame, counts
