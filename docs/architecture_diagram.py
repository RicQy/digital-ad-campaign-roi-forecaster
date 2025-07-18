import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def draw_architecture_diagram():
    fig, ax = plt.subplots(figsize=(12, 8))

    # Draw boxes
    cli_box = mpatches.FancyBboxPatch((0.1, 0.8), 0.8, 0.1, boxstyle="round,pad=0.05", edgecolor='black', facecolor='lightblue')
    data_box = mpatches.FancyBboxPatch((0.1, 0.6), 0.8, 0.1, boxstyle="round,pad=0.05", edgecolor='black', facecolor='lightgreen')
    forecast_box = mpatches.FancyBboxPatch((0.1, 0.4), 0.2, 0.1, boxstyle="round,pad=0.05", edgecolor='black', facecolor='lightcoral')
    optimization_box = mpatches.FancyBboxPatch((0.4, 0.4), 0.2, 0.1, boxstyle="round,pad=0.05", edgecolor='black', facecolor='khaki')
    visualization_box = mpatches.FancyBboxPatch((0.7, 0.4), 0.2, 0.1, boxstyle="round,pad=0.05", edgecolor='black', facecolor='plum')
    config_box = mpatches.FancyBboxPatch((0.1, 0.2), 0.8, 0.1, boxstyle="round,pad=0.05", edgecolor='black', facecolor='wheat')

    # Add boxes to plot
    ax.add_patch(cli_box)
    ax.add_patch(data_box)
    ax.add_patch(forecast_box)
    ax.add_patch(optimization_box)
    ax.add_patch(visualization_box)
    ax.add_patch(config_box)

    # Add text labels
    ax.text(0.5, 0.85, 'CLI Interface', horizontalalignment='center', verticalalignment='center', fontsize=12, weight='bold')
    ax.text(0.5, 0.65, 'Data Processing Layer', horizontalalignment='center', verticalalignment='center', fontsize=12, weight='bold')
    ax.text(0.2, 0.45, 'Forecasting', horizontalalignment='center', verticalalignment='center', fontsize=12)
    ax.text(0.5, 0.45, 'Optimization', horizontalalignment='center', verticalalignment='center', fontsize=12)
    ax.text(0.8, 0.45, 'Visualization', horizontalalignment='center', verticalalignment='center', fontsize=12)
    ax.text(0.5, 0.25, 'Configuration Layer', horizontalalignment='center', verticalalignment='center', fontsize=12, weight='bold')

    # Add arrows
    ax.annotate('', xy=(0.5, 0.8), xytext=(0.5, 0.7), arrowprops=dict(facecolor='black', shrink=0.05))
    ax.annotate('', xy=(0.2, 0.6), xytext=(0.2, 0.5), arrowprops=dict(facecolor='black', shrink=0.05))
    ax.annotate('', xy=(0.5, 0.6), xytext=(0.5, 0.5), arrowprops=dict(facecolor='black', shrink=0.05))
    ax.annotate('', xy=(0.8, 0.6), xytext=(0.8, 0.5), arrowprops=dict(facecolor='black', shrink=0.05))
    ax.annotate('', xy=(0.5, 0.4), xytext=(0.5, 0.3), arrowprops=dict(facecolor='black', shrink=0.05))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.title('Digital Ad Campaign ROI Forecaster - Architecture Diagram', fontsize=14, weight='bold')
    
    # Save to file
    plt.savefig('architecture_diagram.png', dpi=300, bbox_inches='tight')
    print("Architecture diagram saved to 'architecture_diagram.png'")
    plt.show()

if __name__ == "__main__":
    draw_architecture_diagram()
