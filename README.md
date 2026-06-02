# CNN Cats vs Dogs — From Scratch vs Transfert Learning

## Objectif
Comparer un CNN entraîné **from scratch** et un modèle en **transfert learning** (ResNet-18) sur le jeu de données Cats vs Dogs. On mesure l'impact sur la convergence, la performance (accuracy, précision, recall) et la robustesse.

---

## Environnement

```bash
pip install -r requirements.txt
```

**requirements.txt** :
```
torch>=2.0
torchvision>=0.15
matplotlib
numpy
scikit-learn
```

---

## Note sur l'environnement d'exécution

L'entraînement a été initialement tenté sur **Google Colab** (GPU T4), mais les limites de quota GPU gratuites ont rendu l'exécution complète impossible — les sessions se déconnectaient régulièrement et le quota était épuisé avant la fin des entraînements. Le notebook a finalement été exécuté sur **Kaggle** (GPU P100 gratuit, 30h/semaine), qui s'est avéré plus stable et sans interruption pour ce type de tâche.

---

## Organisation des données

1. Télécharger le dataset depuis [Kaggle – Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data)
2. Placer dans le répertoire de travail :

```
Cat_Dog_data/
    train/
        cat/
        dog/
    test/
        cat/
        dog/
```

> Les données ne sont pas sur GitHub (.gitignore).

---

## GPU

Entraînement effectué sur **GPU NVIDIA P100** (Kaggle).
Vérification dans le notebook :
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device utilisé : {device}')  # cuda
```

---

## Commandes pour entraîner

### Expérience A — From Scratch
- Architecture : 3 blocs Conv → BatchNorm → ReLU → MaxPool + classifieur FC
- Batch size : 32 | Epochs : 10 | Dropout : 0.5 | Weight decay : 1e-4
- **SGD** : lr=0.001, momentum=0.9, scheduler StepLR(step=7, γ=0.1)
- **Adam** : lr=1e-4, scheduler CosineAnnealingLR(T_max=10)

### Expérience B — Transfert Learning (ResNet-18)
- Base : ResNet-18 pré-entraîné ImageNet, fine-tuning complet
- Tête : Dropout(0.4) + Linear(512 → 2)
- Batch size : 32 | Epochs : 10
- **SGD** : backbone lr=1e-3, tête lr=1e-2, scheduler StepLR
- **Adam** : backbone lr=1e-4, tête lr=1e-3, scheduler CosineAnnealingLR

---

## Évaluation / Rechargement du modèle

```python
# From Scratch
model_scratch = CatDogCNN(dropout_p=0.5).to(device)
model_scratch.load_state_dict(torch.load('best_scratch_adam.pth', map_location=device))

# Transfert Learning
model_tl = build_resnet18().to(device)
model_tl.load_state_dict(torch.load('best_tl_adam.pth', map_location=device))
```

> Les fichiers `.pth` sont locaux et exclus du dépôt via `.gitignore`.

---

## Résultats

### Tableau comparatif

| Modèle | Optimiseur | Val Acc | Test Acc | Précision | Recall |
|--------|-----------|---------|----------|-----------|--------|
| From Scratch | SGD | 71.22% | — | — | — |
| From Scratch | Adam | 73.40% | 78.36% | 80.22% | 75.28% |
| ResNet-18 TL | SGD | 96.84% | — | — | — |
| ResNet-18 TL | Adam | 96.64% | 98.84% | 98.65% | 99.04% |

### Analyse

**Convergence :** Le transfert learning atteint 94% de validation accuracy dès la première époque, contre seulement 63% pour le CNN from scratch. Cette différence spectaculaire s'explique par les features déjà apprises sur ImageNet — bords, textures, formes — qui sont directement réutilisables pour distinguer chats et chiens. Le from scratch nécessite beaucoup plus d'époques pour extraire des représentations pertinentes depuis zéro.

**Performance :** L'écart de plus de 20 points d'accuracy entre les deux approches (78% vs 98%) illustre la puissance du transfert learning sur des datasets de taille modérée. Le CNN from scratch, malgré la data augmentation et la régularisation (Dropout + BatchNorm), reste limité par la taille du dataset. Le transfert learning bénéficie d'une initialisation riche issue de 1.2 million d'images ImageNet, ce qui lui permet de généraliser bien mieux.

**Optimiseurs :** Pour le from scratch, Adam (73.40%) converge plus vite que SGD (71.22%) grâce à son adaptation automatique du learning rate. Pour le transfert learning, SGD (96.84%) dépasse légèrement Adam (96.64%) sur la validation, mais les deux atteignent des performances excellentes sur le test (~99%). Le learning rate différencié (backbone faible, tête haute) est crucial en transfert learning pour préserver les features pré-apprises tout en adaptant la tête de classification.

---

## Limites & pistes d'amélioration

- Augmenter le nombre d'époques (20+) pour le from scratch afin d'améliorer la convergence.
- Tester des architectures plus légères (MobileNetV3, EfficientNet-B0) pour comparer le ratio performance/paramètres.
- Ajouter du test-time augmentation (TTA) pour améliorer la robustesse en inférence.
- Explorer le gel progressif des couches du transfert learning (layer-wise unfreezing).
- Les limites de GPU gratuites (Colab) ont contraint à réduire le nombre d'époques à 10 au lieu de 20 — des résultats encore meilleurs seraient attendus avec plus de temps d'entraînement.
