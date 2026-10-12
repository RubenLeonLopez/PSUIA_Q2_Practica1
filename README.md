# Búsqueda semántica con FAISS

## Requisitos
- Python 3 (probado con Python 3.14.4 en Ubuntu)
- Librerías indicadas en requirements.txt

## Instalación
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --no-cache-dir -r requirements.txt
```

### Nota sobre PyTorch (versión CPU)
`sentence-transformers` depende de PyTorch. En Linux, pip instala por defecto la versión
con soporte para GPU NVIDIA (CUDA), que añade varios GB de dependencias (`nvidia-*`, `triton`).
Como el proyecto se ejecuta sin GPU, `requirements.txt` usa el repositorio oficial de PyTorch
para CPU:

```text
--extra-index-url https://download.pytorch.org/whl/cpu
torch
```

Así la instalación pasa de más de 5 GB a unos 1–1,5 GB. FAISS también se usa en su
versión CPU (`faiss-cpu`). Para un índice pequeño como el de esta práctica, el
rendimiento en CPU es más que suficiente.

## Ejecución
```bash
python busqueda_faiss.py
```
La primera ejecución descarga el modelo de embeddings (~470 MB).