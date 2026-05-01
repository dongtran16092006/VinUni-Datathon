# Datathon 2026 — Vòng Sơ loại | Đội Googol

## Cấu trúc thư mục

```
.
├── README.md                     # File này
├── tester.py                     # Pipeline chính (single-file, self-contained)
└── data/                         # Thư mục chứa dữ liệu đầu vào
    ├── customers.csv
    ├── orders.csv
    ├── promotions.csv
    ├── sales.csv
    ├── sample_submission.csv
    └── web_traffic.csv
```

## Hướng dẫn tái lập kết quả

### Yêu cầu hệ thống
- Python ≥ 3.9
- Các thư viện: `numpy`, `pandas`, `lightgbm`, `scikit-learn`

### Cài đặt
```bash
pip install numpy pandas lightgbm scikit-learn
```

### Chạy pipeline
```bash
python tester.py
```

Pipeline `tester.py` sẽ tự động:
1. Đọc dữ liệu đầu vào từ thư mục `data/`
2. Trích xuất các tín hiệu ngoại sinh (khuyến mãi, tốc độ tăng trưởng khách hàng)
3. Xây dựng baseline mùa vụ (Seasonal Baseline)
4. Huấn luyện ensemble LightGBM (20 bags, hàm mất mát Tweedie)
5. Dự báo recursive 548 ngày cho giai đoạn (01/2023 – 07/2024)
6. Adaptive blend giữa ML forecast và baseline, sau đó hiệu chuẩn về target mean
7. Xuất file kết quả `submission.csv` tại thư mục gốc

Thời gian chạy: ~5–10 phút (tùy cấu hình CPU). Mã nguồn đã cố định Random seed (`SEED = 42`) để đảm bảo tính tái lập.

## Phương pháp tiếp cận

- **Kiến trúc**: Seasonal Decomposition kết hợp Bagged Gradient Boosting Residual
- **Mô hình**: LightGBM (Tweedie loss, variance_power=1.35), 20 bags
- **Đặc trưng (Features)**: Lag/rolling/EMA autoregressive, calendar features (sin/cos), promo priors, customer growth
- **Đánh giá (Validation)**: Expanding-window holdout trên 300 ngày cuối cùng
- **Tích hợp (Blending)**: Trọng số Time-decay adaptive kết hợp giữa dự báo Machine Learning và Baseline dự báo mùa vụ truyền thống
- **Kiểm soát rủi ro (Drift Guard)**: Áp dụng các ràng buộc chặn trên/dưới cho chuỗi dự báo đệ quy (recursive) dựa trên số liệu median gần nhất để ngăn chặn sự cố error accumulation.

## Ràng buộc tuân thủ

- Không sử dụng Revenue/COGS từ tập test làm đặc trưng
- Không sử dụng dữ liệu bên ngoài
- Đính kèm toàn bộ mã nguồn và dữ liệu đảm bảo kết quả tái lập hoàn toàn
- Seed ngẫu nhiên cố định để triệt tiêu phương sai
