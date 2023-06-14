#!/usr/bin/env python
# coding: utf-8

# In[53]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the base Report class
class Report:
    def __init__(self, dataframe, title, **kwargs):
        self.dataframe = dataframe
        self.title = title
        self.properties = kwargs
        print(self.properties)

    def generate_table(self):
        """This method generates a table using Matplotlib."""
        plt.figure(figsize=(15, 15))
        fig, ax =plt.subplots(1,1)
        ax.axis('tight')
        ax.axis('off')
        self.table = ax.table(cellText=self.dataframe.values,
                              colLabels=self.dataframe.columns,
                              cellLoc = 'center',
                              loc='center')
        self.fig = fig

    def apply_style(self):
        """This method applies common styling to the table."""
        self.table.auto_set_font_size(False)
        self.table.set_fontsize(self.properties.get('font_size', 10))

    def generate(self):
        """This method generates the report by creating and styling the table."""
        self.generate_table()
        self.apply_style()
        self.fig.suptitle(self.title)
        plt.show()


class DataFrameReport(Report):
    """This subclass of Report represents a report with a dataframe."""
    def __init__(self, dataframe, title, **kwargs):
        super().__init__(dataframe, title, **kwargs)
        plt.figure(figsize=(15, 15))

    def apply_style(self):
        """This method applies styling specific to DataFrameReport."""
        super().apply_style()
        # Apply additional or modified styling here


class MatplotlibReport(Report):
    """This subclass of Report represents a report with a matplotlib plot."""
    def __init__(self, dataframe, title, **kwargs):
        super().__init__(dataframe, title, **kwargs)

    def apply_style(self):
        """This method applies styling specific to MatplotlibReport."""
        super().apply_style()
        # Apply additional or modified styling here


class DataFramePlotReport(Report):
    """This subclass of Report represents a report with a dataframe and a plot."""
    def __init__(self, dataframe, title, **kwargs):
        super().__init__(dataframe, title, **kwargs)
    plt.figure(figsize=(10, 5))
    # If no specific styling is needed, no need to override apply_style method.
    # It will use the common styling from the base class.


class ReportFactory:
    """This class generates a report based on a given type."""
    @staticmethod
    def create_report(report_type, dataframe, title, **kwargs):
        if report_type == "dataframe":
            print("Executing DataFrameReport")
            return DataFrameReport(dataframe, title, **kwargs)
        elif report_type == "matplotlib":
            return MatplotlibReport(dataframe, title, **kwargs)
        elif report_type == "dataframe_plot":
            return DataFramePlotReport(dataframe, title, **kwargs)
        else:
            raise ValueError(f"Invalid report_type: {report_type}")





# In[54]:


np.random.seed(42)
data = np.random.randint(low=int(1e6), high=int(1e9), size=(6, 8))
description = ['desc'+str(i) for i in range(1, 7)]
columns = ['description', 't2021Q1', 't2021Q2', 't2021Q3', 't2021Q4', 't2022Q1', 't2022Q2', 't2022Q3', 't2022Q4']
df = pd.DataFrame(data, columns=columns[1:])


# In[70]:


from IPython.display import display, Markdown, HTML
from datetime import datetime

# Create settings dictionary with report parameters
settings = {
    "rai": {
        "report_type": "dataframe",
        "dataframe_key": "df1",
        "title": "Your RAI Report Title",
        "font_size": 10,
        "markdown_text" : f"""
<div align="center">

# RAI Report

This is the Markdown text for the RAI report.
Generated on 

## Section 1
Some content for section 1...

## Section 2
Some content for section 2...

</div>



""",
        # other properties to customize the look and feel of the report
        "markdown_text1": f"<p style='font-family: Arial; font-size: 35px; text-align: center;'>This is the Markdown text for RAI report.<br>Generated on {datetime.now().strftime('%Y-%m-%d')}</p>",
    
    },
    # More settings...
}

# Dictionary to store your dataframes
dataframes = {
    "df1": df,
    
    # Add more dataframes as needed
}

# Generate report for each set of settings
for report_id, report_settings in settings.items():
    # Display the Markdown text above each report
    markdown_text = report_settings.get('markdown_text', '')
    display(Markdown(markdown_text))
    
    # Retrieve the dataframe based on the dataframe_key
    dataframe_key = report_settings.get('dataframe_key')
    if dataframe_key in dataframes:
        # Exclude unnecessary properties before creating the report
        dataframe_settings = {k: v for k, v in report_settings.items() if k not in ['title','dataframe_key', 'markdown_text']}
        
        dataframe = dataframes[dataframe_key]
        # Generate the report
        report = ReportFactory.create_report(dataframe=dataframe, 
                                             title=report_settings['title'], 
                                             **dataframe_settings)
        report.generate()  # generate the report
    else:
        print(f"DataFrame with key '{dataframe_key}' not found.")



# In[28]:


from IPython.display import Markdown

# Create settings dictionary with report parameters
settings = {
    "rai": {
        "report_type": "dataframe",
        "dataframe_key": "df1",
        "title": "Your RAI Report Title",
        "font_size": 10,
        # other properties to customize the look and feel of the report
        "markdown_text": "This is the Markdown text for RAI report.",
    },
    # More settings...
}

# Dictionary to store your dataframes
dataframes = {
    "df1": df1,
    "df2": df2,
    # Add more dataframes as needed
}

# Generate report for each set of settings
for report_id, report_settings in settings.items():
    # Display the Markdown text above each report
    markdown_text = report_settings.get('markdown_text', '')
    display(Markdown(markdown_text))
    
    # Retrieve the dataframe based on the dataframe_key
    dataframe_key = report_settings.get('dataframe_key')
    if dataframe_key in dataframes:
        # Exclude unnecessary properties before creating the report
        dataframe_settings = {k: v for k, v in report_settings.items() if k not in ['report_type','dataframe_key','markdown_text']}
        
        dataframe = dataframes[dataframe_key]
        # Generate the report
        report = ReportFactory.create_report(dataframe=dataframe, 
                                             title=report_settings['title'], 
                                             **dataframe_settings)
        report.generate()  # generate the report
    else:
        print(f"DataFrame with key '{dataframe_key}' not found.")


# In[29]:


from functools import wraps
from typing import Callable

def decorator(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("Before function call")
        result = func(*args, **kwargs)
        print("After function call")
        return result
    return wrapper

@decorator
def my_function():
    """This is my function."""
    print("Inside my_function")

# Call the decorated function
my_function()

# Access the preserved attributes
print(my_function.__name__)  # Output: my_function
print(my_function.__doc__)  # Output: This is my function.


# In[30]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Create a dataframe with one key variable which is categorical and t1-t48 columns time series numeric data
# Key can contain 5 different random categories

# Define the categories
categories = ['Category1', 'Category2', 'Category3', 'Category4', 'Category5']

# Create the 'Key' column
key_column = [random.choice(categories) for _ in range(500)]

# Create the 't1' to 't48' columns
time_series_columns = {f't{i}': np.random.rand(500) for i in range(1, 49)}

# Create the dataframe
df = pd.DataFrame({'Key': key_column, **time_series_columns})

# Introduce some missing values
for col in df.columns:
    if col != 'Key':
        df.loc[df.sample(frac=0.1).index, col] = np.nan

df.head()


# In[31]:


df.shape


# In[32]:


# Generate missing information visualizations based on per key

# Calculate the number of missing values per category
missing_values_per_category = df.groupby('Key').apply(lambda x: x.isnull().sum())

# Plot the missing values
fig, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(missing_values_per_category, ax=ax, cmap='viridis', cbar=False)
plt.title('Missing Values per Category')
plt.xlabel('Time Series Columns')
plt.ylabel('Category')
plt.show()


# In[47]:


missing_values_df = missing_values_per_category.to_frame('Missing Values')


# In[46]:


missing_values_per_category


# In[ ]:





# In[38]:


df_melt


# In[37]:


plt.figure(figsize=(10, 6))
plt.bar(df_melt['time'], df_melt['value'])
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Bar Plot of Time Series Data')
plt.show()


# In[62]:


from IPython.display import display, HTML

column1 = "Data 1"
column2 = "Data 2"
column3 = "Data 3"

table = f"""
<style type="text/css">
div.center {{
  text-align: center;
}}
table.myTable {{ 
  border-collapse: collapse; 
  margin: auto;
}}
table.myTable th {{ 
  background-color: #999; 
  color: white; 
}}
table.myTable td, 
table.myTable th {{ 
  padding: 5px; 
  border: 1px solid #ddd; 
  text-align: center;
}}
</style>

<div class="center">
<table class="myTable">
  <tr>
    <th>Column 1</th>
    <th>Column 2</th> 
    <th>Column 3</th>
  </tr>
  <tr>
    <td>{column1}</td>
    <td>{column2}</td> 
    <td>{column3}</td>
  </tr>
</table>
</div>
"""

display(HTML(table))


# In[65]:


from IPython.display import display, HTML

column1 = "Data 1"
column2 = "Data 2"
column3 = "Data 3"

table = f"""
<style type="text/css">
table.myTable {{ 
  border-collapse: collapse; 
  margin: 0 auto;
  display: block;
}}
table.myTable th {{ 
  background-color: #999; 
  color: white; 
}}
table.myTable td, 
table.myTable th {{ 
  padding: 5px; 
  border: 1px solid #ddd; 
  text-align: center;
}}
</style>

<div style="display: flex; justify-content: center;">
    <table>
      <tr>
        <th>Column 1</th>
        <th>Column 2</th> 
        <th>Column 3</th>
      </tr>
      <tr>
        <td>{column1}</td>
        <td>{column2}</td> 
        <td>{column3}</td>
      </tr>
    </table>
</div>
"""

display(HTML(table))


# <style type="text/css">
# table.myTable {{ 
#   border-collapse: collapse; 
#   margin: 0 auto;
#   display: block;
# }}
# table.myTable th {{ 
#   background-color: #999; 
#   color: white; 
# }}
# table.myTable td, 
# table.myTable th {{ 
#   padding: 5px; 
#   border: 1px solid #ddd; 
#   text-align: center;
# }}
# </style>
# <div style="display: flex; justify-content: center;">
#     <table>
#       <tr>
#         <th>Column 1</th>
#         <th>Column 2</th> 
#         <th>Column 3</th>
#       </tr>
#       <tr>
#         <td>{column1}</td>
#         <td>{column2}</td> 
#         <td>{column3}</td>
#       </tr>
#     </table>
# </div>

# In[ ]:





# <span style="color:red">*This will be red and italic.*</span>
# 

# <div align="center" style="color:#0B2447; font-size:30px; font-weight:bold;">
#     <i>
#         Artificial Intelligence (AI) is a field in computer science that is revolutionizing how we approach problem-solving and task automation across various disciplines. AI aims to mimic human cognitive functions such as learning, problem-solving, perception, and language understanding. It's an umbrella term that encompasses several subfields, including machine learning, where algorithms learn from data, and deep learning, which structures algorithms in layers to create an "artificial neural network" that can learn and make intelligent decisions on its own.
# 
# In recent years, AI has found wide-ranging applications from healthcare, where it aids in disease diagnosis and drug discovery, to finance, where it powers trading algorithms and credit risk assessments. It's integral to technologies we use daily, such as search engines, voice assistants, and recommendation systems. Despite its immense potential, AI also presents significant challenges, including ethical concerns about data privacy, algorithmic bias, and job automation.
# 
# The future of AI holds exciting possibilities. Advancements in AI technologies are set to usher in an era of increased efficiency, productivity, and innovative solutions to complex problems. However, it's essential to move forward with a robust ethical framework and a commitment to mitigating potential adverse effects, ensuring that the benefits of AI are equitably distributed..
#     </i>
# </div>
# 

# In[68]:


import unittest
import pandas as pd
import numpy as np

class TestDataFrame(unittest.TestCase):
    def setUp(self):
        self.expected_df = pd.DataFrame({
            'column1': [1, 2, 3, 4, 5],
            'column2': ['a', 'b', 'c', 'd', 'e']
        })

    def test_dataframe(self):
        # Here is where you would generate your actual DataFrame.
        # This could come from a function or some other part of your code.
        actual_df = pd.DataFrame({
            'column1': [1, 2, 3, 4, 5],
            'column2': ['a', 'b', 'c', 'd', 'e']
        })
        
        pd.testing.assert_frame_equal(self.expected_df, actual_df)
unittest.


# In[73]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from math import ceil


class Report:
    def __init__(self, df, title, rows_per_page=24):
        self.df = df
        self.title = title
        self.rows_per_page = rows_per_page

    def _style_cells(self, table):
        raise NotImplementedError

    def setup_table(self, table):
        table.set_fontsize(14)

    def generate(self):
        num_pages = ceil(len(self.df) / self.rows_per_page)
        for page in range(num_pages):
            start = page * self.rows_per_page
            end = (page + 1) * self.rows_per_page
            fig, ax = plt.subplots(figsize=(20, 6))
            ax.set_title(f'{self.title} (Page {page+1}/{num_pages})')
            table_data = self.df.iloc[start:end]
            table = ax.table(cellText=table_data.values, colLabels=table_data.columns, loc='center')
            self._style_cells(table)
            self.setup_table(table)
            ax.axis('off')
            plt.show()


class DataFrameReport(Report):
    def _style_cells(self, table):
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor('lightseagreen')
                cell.set_text_props(color='white', fontsize=14, ha='center')
            else:
                cell.set_facecolor('lightcyan')
                cell.set_text_props(color='midnightblue', ha='center', alpha=1)
            cell.set_edgecolor('none')


class OrangeTable(Report):
    def _style_cells(self, table):
        num_rows, num_cols = len(self.df)+1, len(self.df.columns)
        for i in range(num_cols-1, -1, -1):
            for j in range(num_rows-1, num_cols - i, -1):
                table[j, i].set_facecolor('lightgrey')


# usage
df = pd.DataFrame(np.random.rand(4, 5), columns=list('ABCDE'))
report = DataFrameReport(df, 'Data Frame Report')
report.generate()

report = OrangeTable(df, 'Orange Table Report')
report.generate()


# In[74]:


import pandas as pd
import matplotlib.pyplot as plt

class Report:
    def __init__(self, df, title, **kwargs):
        self.df = df
        self.title = title
        self.properties = kwargs

    def apply_style(self):
        """Subclasses should override this method to apply custom styling."""
        pass

    def generate(self):
        if self.properties.get('report_type') == 'dataframe':
            self.df.style.set_properties(**self.properties)
        elif self.properties.get('report_type') == 'matplotlib':
            self.df.plot()
            plt.title(self.title)
        elif self.properties.get('report_type') == 'dataframe_plot':
            self.df.plot()
            plt.title(self.title)
            plt.table(cellText=self.df.values, colLabels=self.df.columns, cellLoc='center', loc='bottom')

        self.apply_style()
        plt.show()


class DataFrameReport(Report):
    def apply_style(self):
        self.df.style.set_properties(**{'text-align': 'center'})
        # Add more DataFrame specific styling here


class MatplotlibReport(Report):
    def apply_style(self):
        plt.grid(True)
        # Add more Matplotlib specific styling here


class DataFramePlotReport(Report):
    def apply_style(self):
        plt.grid(True)
        plt.tight_layout()
        # Add more DataFramePlot specific styling here


class ReportFactory:
    @staticmethod
    def create_report(df, title, report_type, **kwargs):
        if report_type == "dataframe":
            return DataFrameReport(df, title, report_type=report_type, **kwargs)
        elif report_type == "matplotlib":
            return MatplotlibReport(df, title, report_type=report_type, **kwargs)
        elif report_type == "dataframe_plot":
            return DataFramePlotReport(df, title, report_type=report_type, **kwargs)
        else:
            raise ValueError(f"Invalid report_type: {report_type}")

# Example usage:
df = pd.DataFrame({'A': range(5), 'B': range(5, 10)})
report = ReportFactory.create_report(df, 'My Title', 'dataframe_plot')
report.generate()


# In[93]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

class Table:
    def __init__(self, df, title, rows_per_page=24):
        self.df = df
        self.title = title
        self.rows_per_page = rows_per_page

    def generate(self):
        num_pages = ceil(len(self.df) / self.rows_per_page)
        for page in range(num_pages):
            start = page * self.rows_per_page
            end = (page + 1) * self.rows_per_page
            fig, ax = plt.subplots(figsize=(20, 6))
            ax.set_title(f'{self.title} (Page {page+1}/{num_pages})')
            table_data = self.df.iloc[start:end]
            table = ax.table(cellText=table_data.values, colLabels=table_data.columns, loc='center')
            self.style_cells(table, ax, table_data)
            ax.axis('off')
            plt.show()

    def style_cells(self, table, ax, table_data):
        # Default style for cells
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor('lightseagreen')
                cell.set_text_props(color='white', fontsize=14,ha='center')
            else:
                cell.set_facecolor('lightcyan')
                cell.set_text_props(color='midnightblue',ha='center',alpha=1)
            cell.set_edgecolor('none')


class OrangeTable(Table):
    def style_cells(self, table, ax, table_data):
        super().style_cells(table, ax, table_data)
        if table_data.shape[1] >= table_data.shape[0]:  # Check if columns >= rows
            num_rows = len(table_data) + 1  # account for header row in matplotlib table
            num_cols = len(self.df.columns)
            for i in range(num_cols-1, -1, -1):
                for j in range(num_rows-1, num_cols - i, -1):
                    
                    table[j, i].set_facecolor('lightgrey')


# In[115]:


settings = {
    'report1': {
        'report_type': 'Table',
        'dataframe_key': 'df1',
        'title': 'Title 1',
        'markdown_text': 'Markdown for report 1',
        'rows_per_page': 30,
    },
    'report2': {
        'report_type': 'OrangeTable',
        'dataframe_key': 'df2',
        'title': 'Title 2',
        'markdown_text': 'Markdown for report 2',
        'rows_per_page': 50,
    },
    'report3': {
        'report_type': 'OrangeTable',
        'dataframe_key': 'df3',
        'title': 'Title 3',
        'markdown_text': 'Markdown for report 3',
        'rows_per_page': 50,
    }

    }


# In[116]:


import pandas as pd
import numpy as np

# Create df1
columns_df1 = ['Metric'] + ['t2021Q1', 't2021Q2', 't2021Q3', 't2021Q4', 't2022Q1', 't2023Q1']
data_df1 = np.random.randint(10000, 100000, size=(6, 6))  # generate random large integers
df1 = pd.DataFrame(data_df1, columns=columns_df1[1:])
df1['Metric'] = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']  # add 'Metric' column
df1 = df1[columns_df1]  # re-order columns

# Create df2
columns_df2 = ['Rank'] + ['t'+str(i+1) for i in range(36)]
data_df2 = np.random.randint(1000, 10000, size=(10, 36))  # generate random integers
df2 = pd.DataFrame(data_df2, columns=columns_df2[1:])
df2['Rank'] = [i+1 for _ in range(1) for i in range(10)]  # add 'Rank' column
df2 = df2[columns_df2]  # re-order columns

# Create df2
columns_df3 = ['Rank'] + ['t'+str(i+1) for i in range(36)]
data_df3 = np.random.randint(1000, 10000, size=(36, 36))  # generate random integers
df3 = pd.DataFrame(data_df3, columns=columns_df3[1:])
df3['Rank'] = [i+1 for _ in range(3) for i in range(12)]  # add 'Rank' column
df3 = df3[columns_df3]  # re-order columns
# Print dataframes
print(df1)
print(df2)
print(df3)



# In[117]:


dataframes = {
    "df1": df1,
    "df2": df2,
    "df3": df3,
    # Add more dataframes as needed
}


# In[118]:


class ReportFactory:
    @staticmethod
    def create_report(report_type, **kwargs):
        if report_type == 'Table':
            return Table(**kwargs)
        elif report_type == 'OrangeTable':
            return OrangeTable(**kwargs)
        else:
            raise ValueError('Invalid report type')

# Generate report for each set of settings
for report_id, report_settings in settings.items():
    # Display the Markdown text above each report
    markdown_text = report_settings.get('markdown_text', '')
    display(Markdown(markdown_text))

    # Retrieve the dataframe based on the dataframe_key
    dataframe_key = report_settings.get('dataframe_key')
    if dataframe_key in dataframes:
        # Exclude unnecessary properties before creating the report
        dataframe_settings = {k: v for k, v in report_settings.items() if k not in ['title','report_type','dataframe_key', 'markdown_text']}
        dataframe = dataframes[dataframe_key]
        
        # Get report type from settings
        report_type = report_settings.get('report_type', 'Table')
        
        # Generate the report
        report = ReportFactory.create_report(report_type, df=dataframe, title=report_settings['title'], **dataframe_settings)
        report.generate()  # generate the report
    else:
        print(f"DataFrame with key '{dataframe_key}' not found.")


# In[ ]:





# In[126]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class Plotter:
    def __init__(self, dataframe, title, plot_type):
        self.dataframe = dataframe
        self.title = title
        self.plot_type = plot_type.lower()

    def plot(self):
        categories = self.dataframe['category'].unique()
        n_categories = len(categories)
        nrows = (n_categories + 1) // 2  # round up division to get number of rows
        fig, axs = plt.subplots(nrows, 2, figsize=(24, 6*nrows))  # 2 plots per row
        
        axs = axs.ravel()  # flatten axs to easily iterate over

        # Add main title
        plt.suptitle(self.title, fontsize=16)

        for i, category in enumerate(categories):
            data = self.dataframe[self.dataframe['category'] == category]

            if self.plot_type == 'line':
                sns.lineplot(data=data, x='time_period', y='value', ax=axs[i])
            elif self.plot_type == 'scatter':
                sns.scatterplot(data=data, x='time_period', y='value', ax=axs[i])
            elif self.plot_type == 'bar':
                sns.barplot(data=data, x='time_period', y='value', ax=axs[i])
            elif self.plot_type == 'area':
                axs[i].fill_between(data['time_period'], data['value'], alpha=0.4)
            else:
                print("Invalid plot type. Choose from 'line', 'scatter', 'bar', or 'area'.")

            axs[i].set_title(f"Category {category}")

        # Remove unused subplots
        if n_categories % 2 != 0:
            fig.delaxes(axs[-1])

        plt.tight_layout()
        plt.show()



# In[179]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class Plotter:
    def __init__(self, dataframe, title, plot_type, color='blue', linestyle='-', marker='o', subplot_shape=(2, None), bg_color=None):
        self.dataframe = dataframe
        self.title = title
        self.plot_type = plot_type.lower()
        self.color = color
        self.linestyle = linestyle
        self.marker = marker
        self.subplot_shape = subplot_shape
        self.bg_color = bg_color

    def plot(self):
        categories = self.dataframe.iloc[:,0].unique()
        n_categories = len(categories)
        nrows = self.subplot_shape[0]
        ncols = self.subplot_shape[1] if self.subplot_shape[1] else (n_categories + nrows - 1) // nrows

        fig, axs = plt.subplots(nrows, ncols, figsize=(6*ncols, 6*nrows))
        axs = axs.ravel()

        if self.bg_color:
            fig.patch.set_facecolor(self.bg_color)

        plt.suptitle(self.title, fontsize=16)

        for i, category in enumerate(categories):
            data = self.dataframe[self.dataframe.iloc[:,0] == category]

            if self.plot_type in ['line', 'scatter', 'bar']:
                if self.plot_type == 'line':
                    plot_func = sns.lineplot
                elif self.plot_type == 'scatter':
                    plot_func = sns.scatterplot
                elif self.plot_type == 'bar':
                    plot_func = sns.barplot

                plot = plot_func(data=data, x=data.columns[1], y=data.columns[2], ax=axs[i], color=self.color)
                axs[i].legend([category])

                if self.plot_type == 'line':
                    line = axs[i].lines[0]
                    line.set_linestyle(self.linestyle)
                    line.set_marker(self.marker)
            elif self.plot_type == 'area':
                axs[i].fill_between(data.iloc[:,1], data.iloc[:,2], alpha=0.4, color=self.color)
                axs[i].legend([category])
            elif self.plot_type == 'heatmap':
                heatmap_data = data.pivot(index=data.columns[1], columns=data.columns[0], values=data.columns[2])
                sns.heatmap(heatmap_data, ax=axs[i], cmap='YlGnBu')
            else:
                print("Invalid plot type. Choose from 'line', 'scatter', 'bar', 'area', or 'heatmap'.")

            axs[i].set_title(f"Category {category}")
            for label in axs[i].get_xticklabels():
                label.set_rotation(45)

        # Remove unused subplots
        for j in range(i+1, nrows*ncols):
            fig.delaxes(axs[j])

        plt.tight_layout()
        plt.show()




# In[180]:


import pandas as pd
import numpy as np

# Categories
categories = list('ABCDEFG')

# Time periods
time_periods = ['2021-Q1', '2021-Q2', '2021-Q3', '2021-Q4', 
                '2022-Q1', '2022-Q2', '2022-Q3', '2022-Q4', '2023-Q1']

# Create empty DataFrame
df = pd.DataFrame(columns=['category', 'time_period', 'value'])

# Populate DataFrame
for cat in categories:
    for period in time_periods:
        df.loc[len(df)] = {'category': cat, 'time_period': period, 'value': np.random.randint(1, 1e6)}

# Check DataFrame
print(df)


# In[181]:


# Instantiate the class
plotter = Plotter(df, 'My Main Title', 'bar', color='tan', linestyle='--', marker='x', subplot_shape=(2, None))

# Plot the graph
plotter.plot()


# In[176]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

class MultiPlotter:
    def __init__(self, dataframe, title, color='blue', linestyle='-', marker='o', subplot_shape=(2, 2), bg_color=None):
        self.dataframe = dataframe
        self.title = title
        self.color = color
        self.linestyle = linestyle
        self.marker = marker
        self.subplot_shape = subplot_shape
        self.bg_color = bg_color

    def plot(self):
        categories = self.dataframe.iloc[:,0].unique()

        fig, axs = plt.subplots(*self.subplot_shape, figsize=(12, 12))
        axs = axs.ravel()

        if self.bg_color:
            fig.patch.set_facecolor(self.bg_color)

        plt.suptitle(self.title, fontsize=16)

        # Line plot
        sns.lineplot(data=self.dataframe, x=self.dataframe.columns[1], y=self.dataframe.columns[2], hue=self.dataframe.columns[0], ax=axs[0])
        axs[0].set_title('Line plot')

        # Stacked Bar plot
        bar_plot_data = self.dataframe.pivot(index=self.dataframe.columns[1], columns=self.dataframe.columns[0], values=self.dataframe.columns[2]).fillna(0)
        bar_plot_data.plot(kind='bar', stacked=True, ax=axs[1])
        axs[1].set_title('Stacked Bar plot')

        # Bubble plot
        scaler = MinMaxScaler()
        self.dataframe['normalized_value'] = scaler.fit_transform(self.dataframe.iloc[:,2].values.reshape(-1,1))
        sns.scatterplot(data=self.dataframe, x=self.dataframe.columns[1], y=self.dataframe.columns[0], hue=self.dataframe.columns[0], size='normalized_value', sizes=(40, 400), ax=axs[2])
        axs[2].set_title('Bubble plot')

        # Heatmap
        heatmap_data = self.dataframe.pivot(index=self.dataframe.columns[0], columns=self.dataframe.columns[1], values=self.dataframe.columns[2])
        sns.heatmap(heatmap_data, ax=axs[3], cmap='YlGnBu')
        axs[3].set_title('Heatmap')

        for ax in axs:
            for label in ax.get_xticklabels():
                label.set_rotation(45)

        plt.tight_layout()
        plt.show()




# In[177]:


plotter = MultiPlotter(df, 'Plots for each category')
plotter.plot()


# In[178]:


class BasePlotter:
    def __init__(self, dataframe):
        if dataframe.shape[1] != 3:
            raise ValueError("Dataframe should have exactly 3 columns.")
        if not pd.to_datetime(dataframe.iloc[:,1], errors='coerce').notna().all():
            raise ValueError("The second column should be convertible to a datetime format.")
        if not pd.to_numeric(dataframe.iloc[:,2], errors='coerce').notna().all():
            raise ValueError("The third column should be numeric.")
        self.dataframe = dataframe

class Plotter(BasePlotter):
    def __init__(self, dataframe, title):
        super().__init__(dataframe)
        self.title = title

class MultiPlotter(BasePlotter):
    def __init__(self, dataframe, title):
        super().__init__(dataframe)
        self.title = title


# In[173]:


def validate_dataframe(func):
    def wrapper(*args, **kwargs):
        dataframe = args[1]  # Assuming dataframe is the second argument
        if dataframe.shape[1] != 3:
            raise ValueError("Dataframe should have exactly 3 columns.")
        if not pd.to_datetime(dataframe.iloc[:,1], errors='coerce').notna().all():
            raise ValueError("The second column should be convertible to a datetime format.")
        if not pd.to_numeric(dataframe.iloc[:,2], errors='coerce').notna().all():
            raise ValueError("The third column should be numeric.")
        return func(*args, **kwargs)
    return wrapper

class Plotter:
    @validate_dataframe
    def __init__(self, dataframe, title):
        self.dataframe = dataframe
        self.title = title

class MultiPlotter:
    @validate_dataframe
    def __init__(self, dataframe, title):
        self.dataframe = dataframe
        self.title = title


# In[ ]:


class Plotter:
    def __init__(self, dataframe, title):
        # Validate the dataframe
        if dataframe.shape[1] != 3:
            raise ValueError("Dataframe should have exactly 3 columns.")
        if not pd.to_datetime(dataframe.iloc[:,1], errors='coerce').notna().all():
            raise ValueError("The second column should be convertible to a datetime format.")
        if not pd.to_numeric(dataframe.iloc[:,2], errors='coerce').notna().all():
            raise ValueError("The third column should be numeric.")
        
        self.dataframe = dataframe
        self.title = title

    def plot(self):
        # Implementation for plotting in the Plotter class
        pass

class MultiPlotter(Plotter):
    def __init__(self, dataframe, title):
        super().__init__(dataframe, title)

    def plot(self):
        # Override the plot method with a new implementation for the MultiPlotter class
        pass

