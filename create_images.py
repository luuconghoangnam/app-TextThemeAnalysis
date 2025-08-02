"""
Script để tạo ảnh placeholder cho README
Chạy script này để tạo các ảnh minh họa mẫu
"""

import matplotlib
matplotlib.use('Agg')  # Sử dụng backend không cần GUI
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import FancyBboxPatch
import os

# Tạo thư mục nếu chưa có
os.makedirs('ImagesREADME', exist_ok=True)

# 1. Tạo ảnh biểu đồ phân phối xác suất
def create_probability_chart():
    categories = ['Business', 'Technology', 'Education', 'Entertainment', 'Sports']
    probabilities = [0.75, 0.15, 0.05, 0.03, 0.02]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart
    bars = ax1.bar(categories, probabilities, color=colors, alpha=0.8)
    ax1.set_title('Phân phối xác suất chủ đề', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Xác suất (%)')
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Thêm giá trị trên từng cột
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    ax2.pie(probabilities, labels=categories, colors=colors, autopct='%1.1f%%',
           startangle=90, explode=(0.1, 0, 0, 0, 0))
    ax2.set_title('Phân phối chủ đề (Pie Chart)', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('ImagesREADME/probability-chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Đã tạo probability-chart.png")

# 2. Tạo ảnh so sánh mô hình
def create_model_comparison():
    models = ['LDA', 'NMF']
    metrics = ['Accuracy', 'Speed', 'Memory Usage', 'Interpretability']
    lda_scores = [0.87, 0.65, 0.70, 0.90]
    nmf_scores = [0.85, 0.80, 0.75, 0.85]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars1 = ax.bar(x - width/2, lda_scores, width, label='LDA', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x + width/2, nmf_scores, width, label='NMF', color='#4ECDC4', alpha=0.8)
    
    ax.set_title('So sánh hiệu suất LDA vs NMF', fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('Điểm số (0-1)', fontsize=14)
    ax.set_xlabel('Metrics', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Thêm giá trị trên từng cột
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('ImagesREADME/model-comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Đã tạo model-comparison.png")

# 3. Tạo ảnh kiến trúc hệ thống
def create_system_architecture():
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Định nghĩa các bước trong pipeline
    steps = [
        ("Input Text", 0.1, 0.8, "#FFE5E5"),
        ("Preprocessing", 0.1, 0.6, "#E5F3FF"),
        ("Vectorization", 0.1, 0.4, "#E5FFE5"),
        ("Model Selection", 0.4, 0.4, "#FFFBE5"),
        ("LDA Model", 0.7, 0.6, "#FFE5F3"),
        ("NMF Model", 0.7, 0.2, "#F3E5FF"),
        ("Prediction", 0.4, 0.1, "#E5FFFF"),
        ("Results", 0.1, 0.1, "#FFF5E5")
    ]
    
    # Vẽ các hộp
    for step, x, y, color in steps:
        box = FancyBboxPatch((x-0.08, y-0.05), 0.16, 0.1, 
                           boxstyle="round,pad=0.01", 
                           facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, step, ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Vẽ các mũi tên
    arrows = [
        ((0.1, 0.75), (0.1, 0.65)),  # Input -> Preprocessing
        ((0.1, 0.55), (0.1, 0.45)),  # Preprocessing -> Vectorization
        ((0.18, 0.4), (0.32, 0.4)),  # Vectorization -> Model Selection
        ((0.48, 0.4), (0.62, 0.55)), # Model Selection -> LDA
        ((0.48, 0.4), (0.62, 0.25)), # Model Selection -> NMF
        ((0.62, 0.6), (0.48, 0.15)), # LDA -> Prediction
        ((0.62, 0.2), (0.48, 0.15)), # NMF -> Prediction
        ((0.32, 0.1), (0.18, 0.1)),  # Prediction -> Results
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))
    
    # Thêm chi tiết preprocessing
    preprocess_details = [
        "• Loại bỏ ký tự đặc biệt",
        "• Chuyển về chữ thường", 
        "• Tách từ (Tokenization)",
        "• Loại bỏ stop words",
        "• Lemmatization"
    ]
    
    detail_text = "\\n".join(preprocess_details)
    ax.text(0.25, 0.6, detail_text, fontsize=9, 
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    ax.set_xlim(0, 0.9)
    ax.set_ylim(0, 0.9)
    ax.set_title('Kiến trúc hệ thống phân tích chủ đề văn bản', 
                fontsize=18, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('ImagesREADME/system-architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Đã tạo system-architecture.png")

# 4. Tạo ảnh workflow
def create_workflow():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Định nghĩa các bước workflow
    workflow_steps = [
        ("1. Mở ứng dụng\\nStreamlit", 0.15, 0.8),
        ("2. Chọn mô hình\\n(LDA/NMF)", 0.5, 0.8),
        ("3. Nhập văn bản\\ncần phân tích", 0.85, 0.8),
        ("4. Nhấn nút\\n'Phân tích'", 0.85, 0.5),
        ("5. Xem kết quả\\nphân loại", 0.5, 0.5),
        ("6. Xem biểu đồ\\nphân phối", 0.15, 0.5),
        ("7. So sánh\\ncác mô hình", 0.15, 0.2),
        ("8. Xuất kết quả\\n(nếu cần)", 0.85, 0.2)
    ]
    
    # Vẽ các bước
    for i, (step, x, y) in enumerate(workflow_steps):
        circle = plt.Circle((x, y), 0.08, color=plt.cm.Set3(i), alpha=0.8)
        ax.add_patch(circle)
        ax.text(x, y, step, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Vẽ các mũi tên kết nối
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)
    ]
    
    for start_idx, end_idx in connections:
        start_x, start_y = workflow_steps[start_idx][1], workflow_steps[start_idx][2]
        end_x, end_y = workflow_steps[end_idx][1], workflow_steps[end_idx][2]
        
        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                   arrowprops=dict(arrowstyle='->', lw=2, color='darkgreen'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Quy trình sử dụng ứng dụng', fontsize=18, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('ImagesREADME/workflow.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Đã tạo workflow.png")

if __name__ == "__main__":
    print("🚀 Bắt đầu tạo ảnh minh họa...")
    
    create_probability_chart()
    create_model_comparison()
    create_system_architecture()
    create_workflow()
    
    print("\\n✅ Hoàn thành! Đã tạo 4 ảnh minh họa trong thư mục ImagesREADME/")
    print("📝 Còn lại các ảnh cần chụp screenshot từ ứng dụng Streamlit:")
    print("   - main-interface.png")
    print("   - text-analysis-demo.png") 
    print("   - usage-guide.png")
    print("   - business-example.png")
    print("   - technology-example.png")
    print("   - project-structure.png")
    print("\\n💡 Hướng dẫn: Mở http://localhost:8501 và chụp ảnh màn hình!")
