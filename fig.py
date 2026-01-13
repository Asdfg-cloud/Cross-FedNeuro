import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

def draw_paper_figure():
    # 创建画布 (宽高比 12:8)
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off') # 隐藏坐标轴

    # --- 颜色定义 ---
    c_srv_bg = '#F0F8FF' # AliceBlue
    c_cli_bg = '#F5F5F5' # WhiteSmoke
    c_blue = '#4682B4'   # SteelBlue
    c_red = '#CD5C5C'    # IndianRed
    c_gray = '#708090'   # SlateGray

    # ================= 1. SERVER LAYER (Top) =================
    # 背景框
    srv_rect = patches.FancyBboxPatch((1, 5.5), 10, 2.2, boxstyle="round,pad=0.1", fc=c_srv_bg, ec='gray', ls='--')
    ax.add_patch(srv_rect)
    ax.text(1.2, 7.5, "Server: Global Knowledge Aggregation", fontsize=12, fontweight='bold', color='black')

    # Public Data (圆柱体模拟)
    ax.add_patch(patches.Ellipse((2, 6.5), 0.8, 0.3, fc='#D3D3D3', ec='black'))
    ax.add_patch(patches.Rectangle((1.6, 6.0), 0.8, 0.5, fc='#D3D3D3', ec='black'))
    ax.add_patch(patches.Ellipse((2, 6.0), 0.8, 0.3, fc='#D3D3D3', ec='black'))
    ax.text(2, 6.25, "Public\nData", ha='center', va='center', fontsize=9)

    # Prototypes
    # Image Proto
    img_proto = patches.FancyBboxPatch((4, 6.2), 2.5, 0.8, boxstyle="round,pad=0.05", fc='white', ec=c_blue, lw=2)
    ax.add_patch(img_proto)
    ax.text(5.25, 6.6, "Global Image\nPrototype", ha='center', va='center', fontsize=10, color=c_blue)

    # Clinical Proto
    cli_proto = patches.FancyBboxPatch((7, 6.2), 2.5, 0.8, boxstyle="round,pad=0.05", fc='white', ec=c_red, lw=2)
    ax.add_patch(cli_proto)
    ax.text(8.25, 6.6, "Global Clinical\nPrototype", ha='center', va='center', fontsize=10, color=c_red)

    # Server Arrows
    ax.annotate("", xy=(4, 6.6), xytext=(2.4, 6.6), arrowprops=dict(arrowstyle="->", color='black'))
    ax.annotate("", xy=(7, 6.6), xytext=(2.4, 6.6), arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", color='black'))

    # ================= 2. BUS LINE (Middle) =================
    ax.plot([0.5, 11.5], [5.2, 5.2], color='#555555', lw=3)
    ax.plot([0.5, 11.5], [5.15, 5.15], color='#555555', lw=3)
    ax.text(11.6, 5.15, "Federated Bus", va='center', fontsize=9, style='italic')

    # Connect Server to Bus
    ax.annotate("", xy=(5.25, 5.2), xytext=(5.25, 6.2), arrowprops=dict(arrowstyle="->", color=c_blue, lw=2))
    ax.annotate("", xy=(8.25, 5.2), xytext=(8.25, 6.2), arrowprops=dict(arrowstyle="->", color=c_red, lw=2))

    # ================= 3. CLIENT LAYER (Bottom) =================

    # --- Client A (Left) ---
    ax.add_patch(patches.FancyBboxPatch((0.5, 0.5), 5, 4.0, boxstyle="round,pad=0.1", fc=c_cli_bg, ec='none'))
    ax.text(0.6, 4.3, "Client A: Multimodal (ResNet)", fontsize=11, fontweight='bold')

    # Pvt Data A
    ax.add_patch(patches.Rectangle((0.8, 2.0), 0.6, 0.8, fc='#D3D3D3', ec='black'))
    ax.text(1.1, 2.4, "Pvt\nData", ha='center', va='center', fontsize=8)

    # Models A
    ax.add_patch(patches.Rectangle((2.0, 3.0), 1.5, 0.6, fc='white', ec=c_blue))
    ax.text(2.75, 3.3, "ResNetFC", ha='center', va='center', fontsize=9)
    # Feature A
    ax.add_patch(patches.Rectangle((3.6, 3.0), 0.3, 0.6, fc='#E0E0E0', ec='black', hatch='///'))

    ax.add_patch(patches.Rectangle((2.0, 1.2), 1.5, 0.6, fc='white', ec=c_red))
    ax.text(2.75, 1.5, "ClinicalNet", ha='center', va='center', fontsize=9)
    # Feature A
    ax.add_patch(patches.Rectangle((3.6, 1.2), 0.3, 0.6, fc='#E0E0E0', ec='black', hatch='///'))

    # Local Loss
    ax.annotate("", xy=(3.75, 2.9), xytext=(3.75, 1.9), arrowprops=dict(arrowstyle="<->", color='red', ls='--'))
    ax.text(3.9, 2.4, "Local\nAlign", fontsize=7, color='red')

    # Flows A
    ax.annotate("", xy=(2.0, 3.3), xytext=(1.4, 2.4), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(2.0, 1.5), xytext=(1.4, 2.4), arrowprops=dict(arrowstyle="->"))


    # --- Client B (Right) ---
    ax.add_patch(patches.FancyBboxPatch((6.5, 0.5), 5, 4.0, boxstyle="round,pad=0.1", fc=c_cli_bg, ec='none'))
    ax.text(6.6, 4.3, "Client B: Unimodal (Transformer)", fontsize=11, fontweight='bold')

    # Pvt Data B
    ax.add_patch(patches.Rectangle((6.8, 2.0), 0.6, 0.8, fc='#D3D3D3', ec='black'))
    ax.text(7.1, 2.4, "Pvt\nData", ha='center', va='center', fontsize=8)

    # Models B
    ax.add_patch(patches.Rectangle((8.0, 2.1), 1.8, 0.6, fc='white', ec=c_blue))
    ax.text(8.9, 2.4, "Transformer", ha='center', va='center', fontsize=9)
    # Feature B
    ax.add_patch(patches.Rectangle((9.9, 2.1), 0.3, 0.6, fc='#E0E0E0', ec='black', hatch='///'))

    # Missing
    ax.add_patch(patches.Rectangle((8.0, 1.0), 1.5, 0.5, fc='none', ec=c_red, ls='--'))
    ax.text(8.75, 1.25, "Missing Clinical", ha='center', va='center', fontsize=8, color=c_red)

    # Flows B
    ax.annotate("", xy=(8.0, 2.4), xytext=(7.4, 2.4), arrowprops=dict(arrowstyle="->"))

    # ================= 4. CONNECTIONS (Lines) =================

    # Distill Lines
    # To A Img
    ax.annotate("", xy=(3.75, 3.7), xytext=(3.75, 5.1), arrowprops=dict(arrowstyle="->", color=c_blue, ls='--'))
    # To A Cli
    ax.annotate("", xy=(3.75, 1.1), xytext=(4.5, 1.1), arrowprops=dict(arrowstyle="<-", color=c_red, ls='--'))
    ax.plot([4.5, 4.5], [1.1, 5.1], color=c_red, ls='--') # Vertical part

    # To B Img (Intra)
    ax.annotate("", xy=(10.05, 2.8), xytext=(10.05, 5.1), arrowprops=dict(arrowstyle="->", color=c_blue, ls='--'))

    # KEY INNOVATION: To B Img (Inter)
    ax.annotate("Knowledge\nTransfer", xy=(10.05, 2.0), xytext=(8.25, 5.1),
                arrowprops=dict(arrowstyle="->", color=c_red, lw=2.5),
                ha='left', va='center', fontsize=9, color=c_red, bbox=dict(boxstyle="round", fc='white', ec=c_red))

    # Save
    plt.tight_layout()
    # plt.show()
    # 直接保存为高清 PDF (矢量图，放论文最好)
    plt.savefig('./Figs/H2_FedNeuro_Framework.pdf', format='pdf', bbox_inches='tight', dpi=300)

    # 或者保存为高清 PNG
    # plt.savefig('H2_FedNeuro_Framework.png', format='png', bbox_inches='tight', dpi=300)

    print("图片已保存到当前目录！")
if __name__ == "__main__":
    draw_paper_figure()