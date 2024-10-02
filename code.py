'''The dataset contains detailed specifications of various laptop models,along with their prices in euros. It is designed to help users understand how different
technical characteristics influence laptop pricing. The data includes information such as brand, model, size, memory, storage, and graphics capabilities, making it useful for market analysis, product comparison, or pricing prediction models.'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

plt.style.use("fivethirtyeight")
%matplotlib inline

laptop_data= pd.read_csv('/kaggle/input/laptop/Laptop.csv')
laptop_data.head()

#Task1:Pricing Analysis
#1.Price Distribution: What is the distribution of prices in the dataset? Are there any outliers or unusual price points?

laptop_data.loc[:5,"Price_euros"]

plt.figure(figsize = (7,5))  
sns.boxplot(x='Company' , y='Price_euros', data= laptop_data)
plt.xlabel("Category")
plt.ylabel("price")
plt.title("laptop_data category wise")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(9,5))
sns.scatterplot(data=laptop_data, x='Company', y='Price_euros')
plt.title('Laptop Price Distribution by Specification')
plt.xlabel('Specification')
plt.ylabel('Price')
plt.xticks(rotation = 45)
plt.show()

plt.figure(figsize=(5,4))
sns.kdeplot(data=laptop_data['Price_euros'], fill=True)
plt.title('price Density plot')
plt.xlabel('Category')
plt.ylabel('Density')
plt.show()

#2.Price Drivers: What are the primary factors that drive the price of a laptop (e.g., brand, screen size, performance)?

plt.figure(figsize=(12, 6))
sns.boxplot(x='Cpu Brand', y='Price_euros', data=laptop_data)
plt.title('Price Distribution by Brand')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Gpu Brand', y='Price_euros', data=laptop_data)
plt.title('Price Distribution by Brand')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(20, 9))
sns.scatterplot(x='Cpu Rate', y='Price_euros', data=laptop_data, hue='Cpu Rate', style='Cpu Rate', s=100)
plt.title('Price vs. Cpu Rate')
plt.xlabel('Cpu Rate')
plt.ylabel('Price')
plt.show() 

plt.figure(figsize=(12, 6))
sns.countplot(x='SSD', data=laptop_data)
plt.title('Count of Laptops by SSD Category')
plt.xlabel('Company')
plt.ylabel('Price')
plt.show()

#3.Price Elasticity: How sensitive is the price of a laptop to changes in its features?

plt.figure(figsize=(20, 9))
sns.scatterplot(x='Ram', y='Price_euros', data=laptop_data,hue='Ram', s=100)
plt.title('Price vs. Ram')
plt.xlabel('Ram')
plt.ylabel('Price')
plt.legend(title='Price vs Ram')
plt.grid()
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Inches', y='Price_euros', data=laptop_data)
plt.title('Price Distribution by screen size')
plt.xlabel('Screen Size')
plt.ylabel('Price')
plt.grid()
plt.show()

plt.figure(figsize=(20, 9))
sns.boxplot(x='Cpu Rate', y='Price_euros', data=laptop_data)
plt.title('Price Distribution by Cpu Rate')
plt.xlabel('Cpu Rate')
plt.ylabel('Price')
plt.grid()
plt.show()

plt.figure(figsize=(12, 6))
sns.violinplot(x='Cpu Brand', y='Price_euros', data=laptop_data, palette='muted')
plt.title('Price Distribution by Cpu Brand')
plt.xlabel('Cpu Brand')
plt.ylabel('Price')
plt.grid()
plt.show()

#Task2:Market Analysis
#1.Market Segmentation: Can the dataset be segmented into different market segments (e.g., budget, mid-range, premium)?

plt.figure(figsize=(10,5))
sns.histplot(laptop_data['Price_euros'],kde=True)
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Density')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Ram', y='Price_euros', data=laptop_data)
plt.title('Price vs. RAM')
plt.xlabel('RAM (GB)')
plt.ylabel('Price (Euros)')
plt.show()

plt.figure(figsize=(22, 9))
sns.boxplot(x='Cpu Model', y='Price_euros', data=laptop_data)
plt.title('Price vs. Processor')
plt.xlabel('Processor')
plt.ylabel('Price (Euros)')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='Company', y='Price_euros', data=laptop_data)
plt.title('Price vs. Brand')
plt.xlabel('Brand')
plt.ylabel('Price (Euros)')
plt.xticks(rotation=45)
plt.show()

quartiles = laptop_data['Price_euros'].quantile([0.25, 0.5, 0.75])
laptop_data['Segment'] = pd.cut(laptop_data['Price_euros'], bins=[0, quartiles[0.25], quartiles[0.5], quartiles[0.75], np.inf],
                         labels=['Budget', 'Mid-Range', 'Premium', 'High-End'])

print(laptop_data['Segment'].value_counts())

plt.figure(figsize=(10, 6))
sns.countplot(x='Segment', data=laptop_data)
plt.title('Market Segmentation')
plt.xlabel('Segment')
plt.ylabel('Count')
plt.show()

#2.Brand Analysis: How do different brands compare in terms of pricing and product offerings?

plt.figure(figsize=(10, 6))
sns.countplot(x='Company', hue='TypeName', data=laptop_data)
plt.title('Product Offerings by Brand')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.legend(title='TypeName')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='Company', y='SSD', data=laptop_data)
plt.title('Average Storage by Brand')
plt.xlabel('Brand')
plt.ylabel('Average Storage (GB)')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='Company', y='Inches', data=laptop_data)
plt.title('Average screen size by Brand')
plt.xlabel('Brand')
plt.ylabel('Average Screen Size')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Company', hue='Gpu Brand', data=laptop_data)
plt.title('GPU Distribution by Brand')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.legend(title='GPU Brand')
plt.xticks(rotation=45)
plt.show()

def clean_ram(x):
    x = str(x).replace("GB", "").strip()  
    try:
        return int(x)
    except ValueError:
        return None 
    
laptop_data['Ram'] = laptop_data['Ram'].apply(clean_ram)

plt.figure(figsize=(10, 6))
sns.boxplot(x='Company', y='Ram', data=laptop_data)
plt.title('RAM Distribution by Brand')
plt.xlabel('Brand')
plt.ylabel('RAM (GB)')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(x='Company', y='Ram', data=laptop_data)
plt.title('RAM Distribution by Brand (Violin Plot)')
plt.xlabel('Brand')
plt.ylabel('RAM (GB)')
plt.xticks(rotation=45)
plt.show()

#3.Trend Analysis: Are there any emerging trends or patterns in the laptop market (e.g., increasing popularity of ultrabooks, growing demand for high-resolution screens)?

plt.figure(figsize=(10, 6))
sns.countplot(x='TypeName', data=laptop_data)
plt.title('Laptop Type Trends')
plt.xlabel('Type')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(6, 6))
laptop_types = laptop_data['TypeName'].value_counts().sort_values(ascending=False)
plt.pie(laptop_types, labels=laptop_types.index, autopct='%1.1f%%', startangle=140)  
plt.title('Laptop Type Trends')
plt.axis('equal')  
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Inches', data=laptop_data)
plt.title('Screen Size Trends')
plt.xlabel('Screen Size (Inches)')
plt.ylabel('Count')
plt.show()

sns.kdeplot(laptop_data['Inches'])
plt.title('Screen Size Distribution (KDE)')
plt.xlabel('Screen Size (Inches)')
plt.ylabel('Density')
plt.show()

plt.figure(figsize=(29, 9))
sns.countplot(x='ScreenResolution', data=laptop_data)
plt.title('Screen Resolution Trends')
plt.xlabel('Screen Resolution')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(26, 9))
sns.countplot(x='Cpu Model', data=laptop_data)
plt.title('Processor Type Trends')
plt.xlabel('Processor Model')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Gpu Brand', data=laptop_data)
plt.title('GPU Type Trends')
plt.xlabel('GPU Brand')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
