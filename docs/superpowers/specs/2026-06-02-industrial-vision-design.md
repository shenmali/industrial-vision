# IndustrialVision — Üretim Hattı Hata Tespiti (CV + PLC)

- **Tarih:** 2026-06-02
- **Durum:** Draft (review için)
- **Tip:** Showcase / vitrin projesi
- **Repo:** `CompVi` (GitHub)

## 1. Özet (TL;DR)

Endüstri 4.0 fabrikalarda kural-bazlı makine görüşü sistemlerinin kırılganlığını
ve yeni defect türlerine karşı yüksek bakım maliyetini azaltmak için,
**kamera görüntüsünden üretim hattı defect tespiti** yapan uçtan uca bir sistem
inşa ediyoruz. Sistem 3-katmanlı bir model pipeline'ı (anomali tespiti +
sınıflandırma + ısı haritası lokalizasyonu) çalıştırır, Jetson Orin gibi edge
donanımda **<50 ms / frame** inference yapar, **Modbus TCP** üzerinden simüle
veya gerçek bir PLC'ye **reject sinyali** gönderir; tüm metrikler Prometheus
+ Grafana ile gözlemlenir.

Bu proje, yazarın vitrin/portfolyo projesidir. Mimari, veri stratejisi, PLC
entegrasyonu ve gözlemlenebilirlik katmanları detaylı biçimde
dokümante edilecek; LinkedIn postu ve blog yazısı bu dokümanlardan
türetilecektir.

## 2. Problem Statement

Geleneksel kural-bazlı makine görüşü (thresholding, edge detection, template
matching) şu sınıflara sahiptir:

- **Yeni defect türlerine karşı kırılgan**: yeni bir çizik şekli ortaya
  çıktığında kural manuel olarak güncellenmelidir.
- **Yüksek false-positive oranı**: aydınlatma değişimleri, kamera jitter'ı
  gibi çevresel faktörler kolaylıkla yanlış alarm üretir.
- **Bakım maliyeti yüksek**: her yeni ürün reçetesi için yeni kurallar
  yazılması gerekir.
- **Sınırlı geri bildirim**: defect lokalizasyonu genelde kaba
  bounding-box'tır, piksel düzeyinde analiz yoktur.

Modern unsupervised anomaly detection modelleri (PatchCore, EfficientAD,
FastFlow) **sadece "good" örneklerle** eğitilir ve üretim hattında
görülmemiş yeni defect türlerini otomatik olarak yakalar. Bu projede bu
modeller endüstriyel bir boru hattının tam ortasına yerleştirilir ve PLC
ile gerçek bir kontrol döngüsü kurulur.

## 3. Hedefler ve Kapsam Dışı

### 3.1 Hedefler (In-Scope)

1. **3-katmanlı hibrit model pipeline'ı**: anomali tespiti, defect sınıflandırma,
   ısı haritası lokalizasyonu.
2. **Hibrit veri stratejisi**: MVTec AD + VisA + kendi webcam verisi, DVC ile
   versiyonlanmış.
3. **Jetson-ready inference**: PyTorch (PC geliştirme) + TensorRT FP16 (Jetson
   deployment) ile asenkron frame pipeline.
4. **Modbus TCP entegrasyonu**: PC'de `pyModbusTCP` ile simüle PLC, opsiyonel
   gerçek PLC desteği (`snap7` / `opcua` adaptörleri ile).
5. **Gözlemlenebilirlik**: Prometheus metrikleri, Grafana dashboard, defect
   event logu.
6. **Vitrin paketi**: README, mimari diyagramlar, demo GIF, benchmark
   tablosu, LinkedIn + blog draft'ları.
7. **Reproducible build**: Docker Compose ile tüm stack tek komutla ayağa
   kalkar (PLC sim + Grafana + Prometheus + app).

### 3.2 Kapsam Dışı (Non-Goals)

- **Gerçek bir fabrikada canlı deployment** (sadece demo / showcase).
- **Çoklu PLC protokolü tek codebase'de** (Modbus TCP MVP, gerçek PLC
  opsiyonel).
- **Cloud training pipeline** (eğitim lokal/PC'de, cloud opsiyonel değil).
- **Multi-tenant / multi-factory** (tek fabrika / tek hat).
- **Active learning / online eğitim** (offline fine-tune yeterli).
- **Birden fazla kamera yönetimi** (tek akış hedefi).

## 4. Başarı Kriterleri (Ölçülebilir)

Vitrin projesi için net, üçüncü tarafın doğrulayabileceği metrikler:

| Kriter | Hedef | Doğrulama Yöntemi |
|--------|-------|--------------------|
| MVTec AD — anomali AUROC | ≥ 0.97 | `notebooks/benchmark.py` |
| MVTec AD — sınıflandırma top-1 | ≥ 0.95 | `notebooks/benchmark.py` |
| Jetson Orin Nano — latency p50 | < 30 ms | `tests/perf/inference_bench.py` |
| Jetson Orin Nano — latency p99 | < 50 ms | `tests/perf/inference_bench.py` |
| Modbus round-trip (Jetson → sim PLC) | < 10 ms | `tests/perf/plc_bench.py` |
| Frame throughput (Jetson) | ≥ 20 FPS | `tests/perf/inference_bench.py` |
| Unit test coverage | ≥ 80% | pytest-cov |
| CI — lint + test + docker build | hepsi yeşil | `.github/workflows/ci.yml` |
| Reproducible demo | tek komut: `docker compose up` | README "Quick Start" |
| Vitrin dokümanları | README + 2 blog taslağı | `docs/` |

## 5. Kullanıcılar ve Paydaşlar

- **Yazar (showcase sahibi)**: GitHub, LinkedIn, blog için referans proje.
- **Mühendisler / teknik değerlendiriciler**: mimari kalite, test coverage,
  CI, kod organizasyonu.
- **Sektör okurları (LinkedIn / blog)**: ROI, gerçek fabrika uyumluluğu,
  modern yaklaşımlar.
- **Potansiyel müşteri / işveren**: "production-ready" iddiasının arkası.

## 6. Sistem Mimarisi (5 Katman)

### 6.1 Genel Bakış

```
┌──────────────────────────────────────────────────────────────────┐
│ L5 — OBSERVABILITY                                                │
│   Prometheus metrics  +  Grafana dashboard  +  alert rules        │
│   (defect_rate, latency p99, modbus_errors, frame_drops)          │
├──────────────────────────────────────────────────────────────────┤
│ L4 — INFERENCE ENGINE                                             │
│   Async pipeline, frame queue, GStreamer / V4L2                   │
│   PC: PyTorch eager  •  Jetson: TensorRT FP16 engine              │
├──────────────────────────────────────────────────────────────────┤
│ L3 — MODEL LAYER (3-katmanlı)                                     │
│   3a. Anomali  : PatchCore + EfficientAD  (AUROC, hot-spot bbox)  │
│   3b. Sınıflandırma: EfficientNet-B0 (defect tipi + confidence)   │
│   3c. Isı haritası : Grad-CAM (piksel düzeyinde lokalizasyon)     │
├──────────────────────────────────────────────────────────────────┤
│ L2 — DATA PLANE                                                   │
│   MVTec AD + VisA + webcam registry  (DVC versioned)              │
│   Augment (Albumentations), stratified splits, leakage check      │
├──────────────────────────────────────────────────────────────────┤
│ L1 — PLC INTEGRATION                                              │
│   Modbus TCP server (pyModbusTCP) + opsiyonel gerçek PLC          │
│   Register map: trigger (DI), reject (DO), confidence (AI)        │
└──────────────────────────────────────────────────────────────────┘
```

### 6.2 Veri Akışı (1 frame)

```
Kamera
  └─► GStreamer / V4L2 capture
        └─► preprocess (resize, normalize)
              └─► L3a Anomali (PatchCore + EfficientAD ensemble)
                    ├─ score < 0.5  → GEÇ (event log: "OK")
                    └─ score ≥ 0.5  ↓
                          └─► L3b Sınıflandırma (EfficientNet-B0)
                                └─► L3c Isı haritası (Grad-CAM)
                                      └─► Severity = f(score, conf, area)
                                            └─► L1 Modbus register yaz
                                                  ├─ 40002 = 1 (reject) veya 0 (ok)
                                                  ├─ 40003 = confidence * 100
                                                  └─ 40010 = defect_code
                                            └─► L5 metric publish
                                                  └─► Grafana event
```

### 6.3 Karar Kuralı (Deterministik)

Bir parça **REJECT** edilir ancak ve ancak:

```
anomaly_score >= 0.5  AND  classifier_confidence >= 0.7
```

Bu eşikler `configs/policy.yaml` üzerinden değiştirilebilir; her frame'in
kararı bu dosyadaki değerlere göre verilir ve versiyonlanır (DVC).

## 7. Bileşen Tasarımı

### 7.1 L2 — Data Plane

**Sorumluluklar:**
- Veri setlerini indirme, doğrulama, normalize etme.
- Train/val/test split'lerini stratified ve leakage-checked üretme.
- Augmentation pipeline'ını deterministic ve versiyonlanabilir tutma.
- DVC ile veri ve split versiyonlama.

**Modüller:**

| Modül | Yol | Sorumluluk |
|-------|-----|------------|
| `registry` | `src/industrial_vision/data/registry.py` | Veri seti indirme, doğrulama, DVC track |
| `splitter` | `src/industrial_vision/data/splitter.py` | Stratified split + group-aware leakage check (kategori bazlı) |
| `augment` | `src/industrial_vision/data/augment.py` | Albumentations pipeline |
| `datasets` | `src/industrial_vision/data/datasets.py` | `AnomalyDataset`, `ClassificationDataset` |

**Veri kaynakları (mix):**

| Kaynak | Kategori sayısı | Image sayısı | Kullanım |
|--------|------------------|---------------|----------|
| MVTec AD | 15 | ~5 000 | Primary benchmark, fine-tune |
| VisA | 12 | ~10 000 | Transfer-learning doğrulaması |
| Custom webcam | 5–10 | ~1 000 | Vitrin "kendi verisi" |

### 7.2 L3 — Model Layer (3-Katmanlı)

#### 7.2.1 Katman 3a — Anomali Tespiti

- **Modeller:** PatchCore (memory-bank) + EfficientAD (student-teacher).
  Ensemble: ağırlıklı ortalama, ağırlıklar val set üzerinde optimize edilir.
- **Çıktı:** `anomaly_score ∈ [0, 1]` + predicted hot-spot bbox.
- **Eğitim:** Sadece `train/good` örnekleri (unsupervised).
- **Metrikler:** AUROC, F1 (threshold 0.5).

#### 7.2.2 Katman 3b — Sınıflandırma

- **Model:** EfficientNet-B0 (ImageNet pretrained), fine-tune.
- **Çıktı:** `defect_type` (kategoriler veri setine göre değişir; MVTec AD için örnek: `scratch, dent, color, contamination, missing_part`) + `confidence ∈ [0, 1]`.
- **Eğitim:** `train/good + train/<defect>` hepsi.
- **Metrikler:** top-1 accuracy, macro-F1, confusion matrix.

#### 7.2.3 Katman 3c — Isı Haritası

- **Yöntem:** Grad-CAM (classifier'ın son conv katmanından).
- **Çıktı:** `(H, W)` heatmap, 0–1 normalize.
- **Metrik:** IoU vs. ground-truth mask (varsa), görsel inceleme.

### 7.3 L4 — Inference Engine

**İki mod:**

| Mod | Platform | Backend | Latency hedefi |
|-----|----------|---------|----------------|
| Geliştirme | PC (CUDA) | PyTorch eager | N/A |
| Üretim | Jetson Orin | TensorRT FP16 | < 50 ms / frame |

**Asenkron tasarım:**
- `capture` thread → `preprocess` thread → `inference` thread → `decision` thread.
- Backpressure: bounded queue (örn. 8 frame) ile drop policy.
- Her thread'in throughput/latency metrikleri L5'e publish.

**GStreamer / V4L2:**
- PC: USB webcam `v4l2src`.
- Jetson: CSI kamera `nvarguscamerasrc`.
- Replay modu: dosyadan frame okuma (CI ve demo için).

### 7.4 L1 — PLC Entegrasyonu

**Modbus TCP, register map:**

| Register | Tip | Anlam |
|----------|-----|-------|
| 40001 (HR) | R/W | Trigger pulse (PC → PLC: "inference tamam") |
| 40002 (HR) | R | Reject decision (0 = ok, 1 = reject) |
| 40003 (HR) | R | Confidence × 100 (uint16, 0–10000) |
| 40004 (HR) | R | Anomaly score × 100 (uint16) |
| 40010 (HR) | R | Defect code (uint16 enum) |
| 00001 (CO) | R | Heartbeat (toggle 1 Hz) |

**Driver soyutlama:**

```python
class PLCClient(Protocol):
    def write_reject(self, decision: Decision) -> None: ...
    def read_trigger(self) -> bool: ...
    def heartbeat(self) -> None: ...
```

Implementasyonlar: `ModbusClient`, opsiyonel `SiemensS7Client` (`snap7`),
opsiyonel `OPCUAClient` (`asyncua`). Config ile seçilir:
`configs/plc.yaml` içinde `driver: pymodbus|opcua|snap7`.

**Simüle PLC:** `deployment/plc_sim/server.py` `pyModbusTCP` ile sahte bir
server çalıştırır. HMI paneli (`deployment/plc_sim/hmi.html`) basit bir
web arayüzüdür: parça sayacı, reject oranı, son defect görüntüsü.

### 7.5 L5 — Observability

**Prometheus metrikleri:**

| Metrik | Tip | Etiketler |
|--------|-----|-----------|
| `iv_frame_latency_seconds` | Histogram | `stage=capture|preprocess|inference|decision` |
| `iv_frame_throughput_fps` | Gauge | — |
| `iv_defect_total` | Counter | `defect_type`, `decision` |
| `iv_modbus_errors_total` | Counter | `op` |
| `iv_modbus_roundtrip_seconds` | Histogram | — |
| `iv_model_score` | Histogram | `model=anomaly|classifier` |

**Grafana dashboard:** `deployment/grafana/dashboards/industrial_vision.json` —
latency heatmap, defect rate, son 100 frame'in skor dağılımı, Modbus health.

**Logging:** Yapılandırılmış JSON log, her defect event'inde frame
snapshot'ı `logs/events/<timestamp>.jpg` olarak kaydedilir.

## 8. Proje Yapısı

```
CompVi/
├── README.md                       # vitrin kapısı
├── LICENSE                         # MIT
├── pyproject.toml                  # uv / poetry
├── docker-compose.yml              # tüm stack
├── Dockerfile
├── .github/
│   └── workflows/
│       ├── ci.yml                  # lint + test + docker build
│       └── benchmark.yml           # opsiyonel nightly benchmark
├── configs/
│   ├── data.yaml
│   ├── model.yaml
│   ├── inference.yaml
│   ├── plc.yaml
│   └── policy.yaml
├── data/                           # DVC tracked
│   ├── raw/
│   │   ├── mvtec_ad/
│   │   ├── visa/
│   │   └── webcam/
│   ├── processed/
│   └── splits/
├── docs/
│   ├── architecture.md             # blog omurgası
│   ├── benchmark.md
│   ├── plc-integration.md
│   ├── blog-post.md
│   ├── linkedin-post.md
│   └── superpowers/
│       ├── specs/                  # bu doküman
│       └── plans/                  # implementation plan
├── deployment/
│   ├── jetson/
│   │   ├── README.md
│   │   ├── tensorrt_export.py
│   │   └── systemd/industrial_vision.service
│   ├── plc_sim/
│   │   ├── server.py
│   │   └── hmi.html
│   ├── prometheus/
│   │   └── prometheus.yml
│   └── grafana/
│       ├── provisioning/
│       └── dashboards/industrial_vision.json
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_train_anomaly.ipynb
│   ├── 03_train_classifier.ipynb
│   ├── 04_benchmark.ipynb
│   └── 05_visualize_predictions.ipynb
├── src/industrial_vision/
│   ├── __init__.py
│   ├── data/
│   │   ├── registry.py
│   │   ├── splitter.py
│   │   ├── augment.py
│   │   └── datasets.py
│   ├── models/
│   │   ├── anomaly/
│   │   │   ├── patchcore.py
│   │   │   ├── efficientad.py
│   │   │   └── ensemble.py
│   │   ├── classifier/
│   │   │   └── efficientnet.py
│   │   ├── heatmap/
│   │   │   └── gradcam.py
│   │   └── registry.py             # model save/load
│   ├── inference/
│   │   ├── pipeline.py
│   │   ├── backends/
│   │   │   ├── pytorch_backend.py
│   │   │   └── tensorrt_backend.py
│   │   ├── capture/
│   │   │   ├── gstreamer.py
│   │   │   ├── v4l2.py
│   │   │   └── file.py
│   │   └── decision.py
│   ├── plc/
│   │   ├── base.py
│   │   ├── pymodbus_client.py
│   │   ├── opcua_client.py
│   │   ├── snap7_client.py         # opsiyonel
│   │   └── factory.py
│   ├── observability/
│   │   ├── metrics.py
│   │   └── logging_config.py
│   ├── api/
│   │   └── fastapi_app.py
│   └── cli.py
└── tests/
    ├── conftest.py
    ├── unit/
    │   ├── test_data_registry.py
    │   ├── test_splitter.py
    │   ├── test_anomaly.py
    │   ├── test_classifier.py
    │   ├── test_gradcam.py
    │   ├── test_plc_clients.py
    │   └── test_decision.py
    ├── integration/
    │   ├── test_pipeline_e2e.py
    │   └── test_plc_roundtrip.py
    └── perf/
        ├── test_inference_bench.py
        └── test_plc_bench.py
```

## 9. Test Stratejisi

### 9.1 Unit testler (pytest)

- Her modülün en az bir happy-path + en az bir failure-path testi.
- Mock kütüphanesi: PLC client (sahte Modbus server), kamera (file capture).
- Coverage hedefi: **≥ %80**.

### 9.2 Integration testler

- **Pipeline E2E**: file capture → 3-katmanlı inference → karar (PLC yazmadan).
- **Modbus roundtrip**: sim PLC + gerçek client, register yaz/ok doğrula.
- CI'da deterministik seed ile tekrarlanabilir.

### 9.3 Performance / Benchmark

- Jetson latency: 1000 frame üzerinden p50/p95/p99, throughput.
- Modbus latency: 1000 write/read üzerinden ortalama, p99.
- Drift guard: regresyon, main'de %10'dan fazla yavaşlama → CI fail.

### 9.4 CI

- Lint: `ruff` + `black --check` + `mypy`.
- Test: `pytest --cov`.
- Docker build: image build + smoke test.
- Benchmark: nightly, regresyon guard.

## 10. Dokümantasyon & Vitrin Çıktıları

| Çıktı | Hedef kitle | İçerik |
|--------|-------------|--------|
| `README.md` | Herkes | Mimari diyagram, demo GIF, quick start, benchmark tablosu, linkler |
| `docs/architecture.md` | Mühendis | 5 katman detay, trade-off'lar, alternatifler |
| `docs/benchmark.md` | Teknik değerlendirici | MVTec AD sonuçları, Jetson latency/power |
| `docs/plc-integration.md` | Sektör okuru | Register map, Modbus, gerçek PLC'ye geçiş |
| `docs/blog-post.md` | Genel okur | 1500 kelime, mimari + ROI + gerçek fabrika senaryoları |
| `docs/linkedin-post.md` | LinkedIn | 1300 karakter, hook + mimari özeti + link |
| Demo GIF | LinkedIn / README | 30 sn canlı demo (kamera → heatmap → PLC reject ışığı) |

## 11. Zaman Planı (Milestones)

| # | Milestone | Çıktı |
|---|-----------|--------|
| M0 | Repo scaffold + CI + lint | pyproject, ruff, github actions, docker-compose skeleton |
| M1 | Data plane + datasets | DVC pipeline, splits, datasets.py, registry indirme |
| M2 | Anomali modeli (PatchCore + EfficientAD) | Train, val, eval notebooks, AUROC ≥ 0.97 |
| M3 | Classifier (EfficientNet-B0) | Fine-tune, confusion matrix, top-1 ≥ 0.95 |
| M4 | Heatmap (Grad-CAM) | Görsel çıktı, IoU raporu |
| M5 | Inference engine (PC) | Async pipeline, replay mod, E2E test |
| M6 | PLC entegrasyonu | Modbus client + sim server + HMI, round-trip test |
| M7 | Observability | Prometheus, Grafana, alert kuralları |
| M8 | Jetson deploy | TensorRT export, perf test, systemd service |
| M9 | Vitrin paketi | README, demo GIF, blog + LinkedIn draft, benchmark |
| M10 | Polish + release | Tag v1.0.0, GitHub release, post yayınla |

Toplam ~10 milestone, tahmini 4–6 hafta solo çalışma (hafta içi akşam +
hafta sonu 1–2 gün).

## 12. Riskler ve Azaltımlar

| Risk | Olasılık | Etki | Azaltma |
|------|----------|------|---------|
| MVTec AD indirme / lisans sorunu | Düşük | Orta | Mirror'lar, alternatif VisA, sentetik üretim |
| Jetson latency hedefi tutmaz | Orta | Yüksek | Model pruning/quant, INT8 export, batch=1 optimize |
| PatchCore bellek footprint | Düşük | Orta | Core-set subsampling, %1–%10 bank |
| Modbus gerçek PLC uyumsuzluğu | Orta | Düşük | Driver soyutlama + adapter, demo sim PLC |
| Webcam verisi kalitesi | Yüksek | Orta | Işık/kontrast standardizasyonu, sentetik augment |
| Scope creep | Yüksek | Yüksek | Strict milestones, her sprint sonunda vitrin çıktısı |
| Test coverage düşer | Orta | Orta | CI gate, coverage raporu her PR'da |

## 13. Açık Sorular

Yok — tüm kararlar bu spec'te belirlenmiştir. İleride çıkabilecek ek
sorular `docs/superpowers/specs/2026-06-02-questions.md` altında
toplanır.
