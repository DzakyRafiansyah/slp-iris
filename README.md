````markdown
# slp-iris-sigmoid-mse

> Single Layer Perceptron (SLP) untuk **Iris Setosa (0)** vs **Iris Versicolor (1)** dengan **Sigmoid + MSE (delta rule)** — mereplikasi metode di Google Sheet **5 epoch, 100 langkah/epoch, urutan tetap**.

---

## Fitur Utama

- **Aktivasi**: Sigmoid, **Loss**: MSE (delta rule)
- **Inisialisasi**: θ₀..θ₄ = 0.5 (bias + 4 bobot)
- **Training**: 5 epoch × 100 langkah/epoch, **urutan data tetap** (tanpa shuffle)

---

## Struktur Input

- **CSV**: minimal kolom `x1, x2, x3, x4` + salah satu `y | target | species`  
  (Bila `species` ada, mapping seperti di atas). Kolom tambahan `x0` (bias) **diabaikan**.

---

## Instalasi Cepat

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install numpy pandas openpyxl
```
````

---

## Cara Pakai

### Pakai CSV (mis. `iris_train.csv`)

```bash
python slp_iris_sigmoid.py \
  --csv "./iris_train.csv" \
  --epochs 5 --eta 0.1 \
  --metric_mode streaming \
  --out slp_results.csv --print_params
```

---

## Format Output (`slp_results.csv`)

Kolom: `epoch, train_acc, train_mse, val_acc, val_mse, w, b`

- `w` = daftar 4 bobot (θ₁..θ₄), `b` = bias (θ₀) pada **akhir epoch** ke-`epoch`.

## Struktur Repo

```
.
├── slp_iris_sigmoid.py
├── iris_train.csv                
├── slp_results.csv               # output per-epoch
└── README.md
```
