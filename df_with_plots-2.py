#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# In[2]:


np.random.seed(42)
data = np.random.randint(low=int(1e6), high=int(1e9), size=(6, 8))
description = ['desc'+str(i) for i in range(1, 7)]
columns = ['description', 't2021Q1', 't2021Q2', 't2021Q3', 't2021Q4', 't2022Q1', 't2022Q2', 't2022Q3', 't2022Q4']
df = pd.DataFrame(data, columns=columns[1:])
df.insert(0, 'description', description)

# Transpose the dataframe
df_T = df.set_index('description').T


# In[3]:


df_T


# In[12]:


# Define a function to smooth data
def smooth_data(x, y):
    # Create a cubic spline interpolation
    interp = interp1d(x, y, kind='cubic')
    # Use more points for a smoother plot
    xnew = np.linspace(x.min(), x.max(), 500)
    ynew = interp(xnew)
    return xnew, ynew

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Plot each series
for i, column in enumerate(df_T.columns):
    ax = axs[i//3, i%3]
    y = df_T[column].values
    x = np.arange(len(y))
    xnew, ynew = smooth_data(x, y)
    ax.plot(xnew, ynew, label=column)
    ax.set_title(column)
    ax.set_xlabel('Quarters')
    ax.set_ylabel('Values')

# Adjust layout for better visibility
plt.tight_layout()
plt.show()




# In[17]:


from scipy.interpolate import interp1d
import six

# Define a function to smooth data
def smooth_data(x, y):
    # Create a cubic spline interpolation
    interp = interp1d(x, y, kind='cubic')
    # Use more points for a smoother plot
    xnew = np.linspace(x.min(), x.max(), 500)
    ynew = interp(xnew)
    return xnew, ynew

def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            if k[1] == 0:  # Add this condition to check if the cell is in the first column
                cell.set_facecolor('#ADD8E6')
            else:
                cell.set_facecolor(row_colors[k[0]%len(row_colors)])
    ax.set_title('Data Table', pad=20)
    return ax

render_mpl_table(df, header_columns=0, col_width=2.0)

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Plot each series
for i, column in enumerate(df_T.columns):
    ax = axs[i//3, i%3]
    y = df_T[column].values
    x = np.arange(len(y))
    xnew, ynew = smooth_data(x, y)
    ax.plot(xnew, ynew, label=column)
    ax.set_title(column)
    ax.set_xlabel('Quarters')
    ax.set_ylabel('Values')

fig.suptitle('My Common Title', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
plt.show()




# In[24]:


from scipy.interpolate import interp1d
import six

# Define a function to smooth data
def smooth_data(x, y):
    # Create a cubic spline interpolation
    interp = interp1d(x, y, kind='cubic')
    # Use more points for a smoother plot
    xnew = np.linspace(x.min(), x.max(), 500)
    ynew = interp(xnew)
    return xnew, ynew
header_colors = ['blue']
first_column_color = 'lightblue'
cell_colors = ['white', 'gainsboro']
def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            if k[1] == 0:  # Add this condition to check if the cell is in the first column
                cell.set_facecolor('#ADD8E6')
            else:
                cell.set_facecolor(row_colors[k[0]%len(row_colors)])
    ax.set_title('Data Table', pad=20)
    return ax

render_mpl_table(df, header_columns=0, col_width=2.0)

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Plot each series
for i, column in enumerate(df_T.columns):
    ax = axs[i//3, i%3]
    y = df_T[column].values
    x = np.arange(len(y))
    xnew, ynew = smooth_data(x, y)
    ax.plot(xnew, ynew, label=column)
    ax.set_title(column)
    ax.set_xlabel('Quarters')
    ax.set_ylabel('Values')

fig.suptitle('My Common Title', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
plt.show()




# In[12]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Generate some sample data
np.random.seed(0)  # For reproducibility
data = {f't{i}': np.random.randint(1, 100, 10) for i in range(1, 37)}
df = pd.DataFrame(data)
df['rank'] = range(1, 11)  # rank from 1 to 10

# Reset index so 'rank' becomes a column
df = df.reset_index(drop=True)



df


# In[13]:


# Melt the DataFrame
df_long = df.melt(id_vars='rank', var_name='time', value_name='no_of_accounts')

# Now, df_long should have three columns: 'rank', 'time', and 'no_of_accounts'

# Get unique ranks
ranks = df_long['rank'].unique()


# In[8]:


df_long


# In[9]:


# Create subplots in a grid of 3 columns
rows = len(ranks) // 3 + (len(ranks) % 3 != 0)
fig, axs = plt.subplots(rows, 3, figsize=(15, 5 * rows))

# Adjust axs to be a 1D array for easier iteration
axs = axs.flatten()

for i, rank in enumerate(ranks):
    df_rank = df_long[df_long['rank'] == rank]
    axs[i].plot(df_rank['time'], df_rank['no_of_accounts'], label=f'Rank {rank}')
    axs[i].set_xlabel('Time Period')
    axs[i].set_ylabel('Number of Accounts')
    axs[i].legend(loc='upper left')

# Remove unused subplots
for i in range(len(ranks), len(axs)):
    fig.delaxes(axs[i])

plt.tight_layout()
plt.show()


# In[2]:


import pandas as pd
import numpy as np

# Set a random seed for reproducibility
np.random.seed(0)

# Generate a DataFrame
months = pd.date_range(start='1/1/2020', periods=36, freq='M')
ranks = range(1, 11)

data = {
    'Month': np.repeat(months, len(ranks)),
    'FICO_rank': np.tile(ranks, len(months)),
    'Total_acts': np.random.randint(low=100, high=500, size=len(months)*len(ranks)),
}

df = pd.DataFrame(data)

# Show the first few rows of the dataframe
print(df.shape)


# In[3]:


import seaborn as sns
import matplotlib.pyplot as plt

# First, reshape your dataframe to have FICO ranks as index, months as columns, and total acts as values
heatmap_data = df.pivot(index='FICO_rank', columns='Month', values='Total_acts')

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, cmap="YlGnBu")

plt.title('Total acts by FICO rank over time')
plt.show()


# In[4]:


# First, reshape your dataframe to have months as index, FICO ranks as columns, and total acts as values
area_data = df.pivot(index='Month', columns='FICO_rank', values='Total_acts')

# Plot the stacked area chart
area_data.plot(kind='area', stacked=True)

plt.title('Total acts by FICO rank over time')
plt.xlabel('Month')
plt.ylabel('Total acts')
plt.show()


# In[6]:


import pandas as pd
import numpy as np

# Set a random seed for reproducibility
np.random.seed(0)

# Generate a DataFrame
ranks = range(1, 11)
columns = ['Rank'] + [f'T{i}' for i in range(1, 37)]
data = np.random.randint(low=100, high=500, size=(len(ranks), len(columns)-1))
data = np.column_stack([ranks, data])

df = pd.DataFrame(data, columns=columns)

# Show the first few rows of the dataframe
print(df)


# In[7]:


df_melted = df.melt(id_vars='Rank', var_name='Month', value_name='Total_acts')

# Show the first few rows of the reshaped dataframe
print(df_melted.head())


# In[9]:


# Convert Month column to integer after removing 'T' character
df_melted['Month'] = df_melted['Month'].str.replace('T', '').astype(int)

# Sort DataFrame by Month
df_melted.sort_values(by='Month', inplace=True)

# Reset index
df_melted.reset_index(drop=True, inplace=True)


# In[10]:


import seaborn as sns
import matplotlib.pyplot as plt

# First, reshape your dataframe to have Rank as index, Month as columns, and Total acts as values
heatmap_data = df_melted.pivot(index='Rank', columns='Month', values='Total_acts')

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, cmap="YlGnBu")

plt.title('Total acts by Rank over time')
plt.show()


# In[11]:


area_data = df_melted.pivot(index='Month', columns='Rank', values='Total_acts')
area_data.plot(kind='area', stacked=True)
plt.title('Total acts by Rank over time')
plt.xlabel('Month')
plt.ylabel('Total acts')
plt.show()


# In[12]:


# Plot a line for each FICO rank
plt.figure(figsize=(10, 6))
for rank in df_melted['Rank'].unique():
    data = df_melted[df_melted['Rank'] == rank]
    plt.plot(data['Month'], data['Total_acts'], label=f'Rank {rank}')

plt.xlabel('Month')
plt.ylabel('Total Acts')
plt.legend()
plt.title('Total Acts for each FICO Rank over Time')
plt.show()


# In[21]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='Month', y='Total_acts', data=df_melted)
plt.xlabel('Month')
plt.ylabel('FICO Rank')
plt.title('Distribution of Total Acts by FICO Rank')
plt.show()


# In[14]:


g = sns.FacetGrid(df_melted, col="Month", col_wrap=6, height=4, aspect=1)
g = g.map(sns.boxplot, "Rank", "Total_acts", order=sorted(df_melted['Rank'].unique()))


# In[15]:


plt.figure(figsize=(10, 6))
sns.violinplot(x='Rank', y='Total_acts', data=df_melted)
plt.xlabel('FICO Rank')
plt.ylabel('Total Acts')
plt.title('Distribution of Total Acts by FICO Rank')
plt.show()


# In[16]:


sns.pairplot(df_melted, hue='Rank')
plt.show()


# In[23]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

# Generate scatter plot with color coding based on 'Rank'
scatter = plt.scatter(df_melted['Month'], df_melted['Rank'], s=df_melted['Total_acts'],
                      c=df_melted['Rank'], alpha=0.5, cmap='viridis')

plt.xlabel('Month', fontsize=14)
plt.ylabel('Rank', fontsize=14)
plt.title('Total acts by Rank and Month', fontsize=16)

# Add a color bar on the right side of the scatterplot
plt.colorbar(scatter)

# Adding gridlines
plt.grid(False)

plt.show()


# In[24]:


fig = plt.figure(figsize=(15, 20))  # Adjust size as needed

ranks = df_melted['Rank'].unique()

n = len(ranks)  # number of unique ranks
cols = 2  # let's say we want 2 columns of subplots
rows = n // cols + (n % cols > 0)  # calculate number of rows needed

for i, rank in enumerate(ranks, 1):
    data = df_melted[df_melted['Rank'] == rank]
    
    ax = plt.subplot2grid((rows, cols), ((i-1)//cols, (i-1)%cols))
    ax.plot(data['Month'], data['Total_acts'], label=f'Rank {rank}')
    
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Acts')
    ax.legend()
    ax.set_title(f'Total Acts for Rank {rank} over Time')

    # set the face color of the subplot
    ax.set_facecolor('lightgray')

fig.tight_layout()  # adjust spacing between subplots
plt.show()


# In[32]:


import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm

class AdvancedPlotter:
    def __init__(self, df):
        self.df = df

    def create_plots(self, filename):
        with PdfPages(filename) as pdf:
            fig = plt.figure(figsize=(20, 20))  # Adjust size as needed

            # 1. Scatter plot
            ax1 = plt.subplot2grid((2, 2), (0, 0), fig=fig)
            ranks = self.df['Rank'].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(ranks)))

            for i, rank in enumerate(ranks):
                data = self.df[self.df['Rank'] == rank]
                ax1.scatter(data['Month'], data['Rank'], s=data['Total_acts'], c=colors[i], label=f'Rank {rank}')

            ax1.set_xlabel('Month')
            ax1.set_ylabel('Rank')
            ax1.set_title('Total acts by Rank and Month')
            ax1.legend()

            # 2. Stacked plot
            ax2 = plt.subplot2grid((2, 2), (0, 1), fig=fig)
            df_pivot = self.df.pivot(index='Month', columns='Rank', values='Total_acts').fillna(0)
            df_pivot.plot(kind='bar', stacked=True, ax=ax2)
            ax2.set_xlabel('Month')
            ax2.set_ylabel('Total Acts')
            ax2.set_title('Stacked Bar Plot of Total Acts by Month and Rank')

            # 3. Heatmap
            ax3 = plt.subplot2grid((2, 2), (1, 0), fig=fig)
            sns.heatmap(df_pivot, cmap='viridis', ax=ax3)
            ax3.set_xlabel('Rank')
            ax3.set_ylabel('Month')
            ax3.set_title('Heatmap of Total Acts by Month and Rank')

            # 4. Box plot
            ax4 = plt.subplot2grid((2, 2), (1, 1), fig=fig)
            sns.boxplot(x='Rank', y='Total_acts', data=self.df, ax=ax4)
            ax4.set_xlabel('Rank')
            ax4.set_ylabel('Total Acts')
            ax4.set_title('Box Plot of Total Acts by Rank')

            fig.tight_layout()  # adjust spacing between subplots
            pdf.savefig(fig)
            plt.close(fig)


# In[33]:


plotter = AdvancedPlotter(df_melted)
plotter.create_plots('advanced_plots.pdf')


# In[31]:


import matplotlib.cm as cm

# Create a colormap based on the number of unique ranks
cmap = cm.get_cmap('tab10', len(df_melted['Rank'].unique()))

plt.figure(figsize=(10, 6))

for i, rank in enumerate(df_melted['Rank'].unique()):
    data = df_melted[df_melted['Rank'] == rank]
    plt.scatter(data['Month'], data['Rank'], s=data['Total_acts'], 
                c=cmap(i), label=f'Rank {rank}')

plt.xlabel('Month')
plt.ylabel('Rank')
plt.title('Total acts by Rank and Month')
plt.legend()
plt.show()



# In[ ]:




