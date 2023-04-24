import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages

def visualize_dataframe_to_pdf(df, title, filename, figsize=(12, 6)):
    # Set up the subplot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    
    # Define header color and text color
    header_color = mcolors.CSS4_COLORS['steelblue']
    header_text_color = 'white'
    fig.suptitle(title, fontsize=16, y=0.95)
    
    # Get all the columns in the dataframe
    subset_columns = df.columns

    # Create cell colors array
    cell_colors = np.full((df.shape[0] + 1, len(subset_columns)), 'white')
    cell_colors[0, :] = header_color

    # Create cell text colors array
    cell_text_colors = np.full((df.shape[0] + 1, len(subset_columns)), 'black')
    cell_text_colors[0, :] = header_text_color

    # Create the table
    table = ax.table(cellText=np.vstack([subset_columns, df[subset_columns].values]), cellLoc='center', loc='center', cellColours=cell_colors)

    # Apply text colors and bold to the table
    for cell_coordinates, cell in table._cells.items():
        r, c = cell_coordinates
        cell.set_text_props(weight='bold' if r == 0 else 'normal', color=cell_text_colors[r, c], fontsize=14 if r == 0 else 7)
        cell.set_edgecolor('gray')
        cell.set_linewidth(0.5)

    ax.axis('off')

    # Tighten layout
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.00, top=0.96)
    
    # Save the plot to a PDF file
    with PdfPages(filename) as pdf:
        pdf.savefig(fig)

    # Close the figure to release memory
    plt.close(fig)
