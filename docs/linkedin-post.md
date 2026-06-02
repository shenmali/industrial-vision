# LinkedIn Post (Draft)

🚀 Just shipped my new showcase project: IndustrialVision — an end-to-end
defect detection system for production lines.

The pitch: traditional rule-based machine vision is brittle and expensive
to maintain. A modern CV stack can do better.

The stack:
• PatchCore + EfficientAD ensemble for unsupervised anomaly detection
• EfficientNet-B0 for defect type classification
• Grad-CAM for pixel-level localization
• Jetson Orin + TensorRT FP16 for sub-50ms inference
• Modbus TCP to a real (or simulated) PLC for reject decisions
• Prometheus + Grafana for observability
• MVTec AD + VisA + my own webcam data, all DVC-tracked

Everything is reproducible: `docker compose up` brings up the PLC sim,
Prometheus, Grafana, and the inference service.

📂 GitHub: github.com/<you>/CompVi
📝 Architecture: docs/architecture.md
🎥 30-sec demo: docs/demo.gif (TBD)

What would you add? Active learning? Multi-camera? Cloud training?

#ComputerVision #Industry40 #ManufacturingAI #PyTorch #EdgeAI
