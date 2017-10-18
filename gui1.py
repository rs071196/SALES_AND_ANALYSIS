# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 09:40:34 2017

@author: admin
"""
from __future__ import division
from collections import Counter
from linear_algebra import distance, vector_subtract, scalar_multiply
import numpy as np 
import pandas as pd 
from pandas import Series


from collections import Counter
import math, random

from sklearn.cluster import KMeans
from sklearn import preprocessing,cross_validation
from matplotlib import style

import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("vgsales.csv")
from tkinter import *

root = Tk()

ButtonImage = PhotoImage(file='sales.png')
testButton = Button(root, image=ButtonImage)
testButton.pack()

def datainfo():
        print(df.head())
        print("*****************************************************************")
        df.info()
        print("*****************************************************************")
button2=Button(root,text="data info.",command=datainfo,padx=30,pady=15)
button2.pack(side=LEFT)

def analys():
    platGenre = pd.crosstab(df.Platform,df.Genre)
    platGenreTotal = platGenre.sum(axis=1).sort_values(ascending = False)
    plt.figure(figsize=(8,6))
    sns.barplot(y = platGenreTotal.index, x = platGenreTotal.values, orient='h')
    plt.ylabel = "Platform"
    plt.xlabel = "The amount of games"
    plt.show()
    print("\nYou can see DS and PS2 have the most games in their platform \n*****************************************************************")

button1=Button(root,text="Analysis",command=analys,padx=30,pady=15)
button1.pack(side=LEFT)

def genre():
     platGenre = pd.crosstab(df.Platform,df.Genre)
     platGenre['Total'] = platGenre.sum(axis=1)
     popPlatform = platGenre[platGenre['Total']>1000].sort_values(by='Total', ascending = False)
     neededdata = popPlatform.loc[:,:'Strategy']
     maxi = neededdata.values.max()
     mini = neededdata.values.min()
     popPlatformfinal = popPlatform.append(pd.DataFrame(popPlatform.sum(), columns=['total']).T, ignore_index=False)
     sns.set(font_scale=0.7)
     plt.figure(figsize=(10,5))
     sns.heatmap(popPlatformfinal, vmin = mini, vmax = maxi, annot=True, fmt="d")
     plt.xticks(rotation = 90)
     plt.show()
        
     print("\nSo, you can see the popular genre game of DS, PS2 and PS3.\n\n")
     print("*****************************************************************")
a=Button(root,text="GAME DETAILS HAVING PLATFORM MORE THAN 1000",command= genre, justify=LEFT,padx=30,pady=15)
a.pack(side=LEFT)

def pred():
     GenreGroup = df.groupby(['Genre']).sum().loc[:, 'NA_Sales':'Global_Sales']
     GenreGroup['NA_Sales%'] = GenreGroup['NA_Sales']/GenreGroup['Global_Sales']
     GenreGroup['EU_Sales%'] = GenreGroup['EU_Sales']/GenreGroup['Global_Sales']
     GenreGroup['JP_Sales%'] = GenreGroup['JP_Sales']/GenreGroup['Global_Sales']
     GenreGroup['Other_Sales%'] = GenreGroup['Other_Sales']/GenreGroup['Global_Sales']
     plt.figure(figsize=(8, 10))
     sns.set(font_scale=0.7)
     plt.subplot(211)
     sns.heatmap(GenreGroup.loc[:, 'NA_Sales':'Other_Sales'], annot=True, fmt = '.1f')
     plt.title("Comparation of each area in each Genre")
     plt.subplot(212)
     sns.heatmap(GenreGroup.loc[:,'NA_Sales%':'Other_Sales%'], vmax =1, vmin=0, annot=True, fmt = '.2%')
     plt.title("Comparation of each area in each Genre(Pencentage)")
     plt.show()
     print("*****************************************************************")
        

c=Button(root,text="\nGENRE OF GAMES SALES",command= pred,padx=30,pady=7)
c.pack(side=LEFT)

def analys1():
    series = Series.from_csv('vgsales.csv')
    print(series.describe())
b1=Button(root,text="STATISTICS ANALYSIS",command= analys1,padx=30,pady=15)
b1.pack(side=LEFT)

def genre1():
    style.use('ggplot')
    data=pd.read_excel("vgsales.xlsx")
    data.drop([],1,inplace = True)
    data.convert_objects(convert_numeric=True)
    data.fillna(0,inplace=True)


    def handle_non_numerical_data(data):
        columns=data.columns.values
        for column in columns:
            text_digit_vals={}# builting a dictionary
            def convert_to_int(val):# handle non numeric data
                return text_digit_vals[val]
        
            if data[column].dtype!=np.int64 and data[column].dtype!=np.float64:
                column_contents=data[column].values.tolist()# built a list of non int and non float column 
                unique_elements=set(column_contents)
                x = 0
                for unique in unique_elements:
                    if unique not in text_digit_vals:
                        text_digit_vals[unique]=x
                        x+=1
            
                data[column]= list(map(convert_to_int,data[column]))
                
        return data
    data= handle_non_numerical_data(data)
    print(data.head())

a1=Button(root,text="DATA CONVERSION",command= genre1, justify=LEFT,padx=30,pady=15)
a1.pack(side=LEFT)

def ana():
    
        def random_kid():
            return random.choice(["boy", "girl"])

        def uniform_pdf(x):
            return 1 if x >= 0 and x < 1 else 0

        def uniform_cdf(x):
            if x < 0:   return 0    # uniform random is never less than 0
            elif x < 1: return x    # e.g. P(X < 0.4) = 0.4
            else:       return 1    # uniform random is always less than 1

        def normal_pdf(x, mu=0, sigma=1):
            sqrt_two_pi = math.sqrt(2 * math.pi)
            return (math.exp(-(x-mu) ** 2 / 2 / sigma ** 2) / (sqrt_two_pi * sigma))

        def plot_normal_pdfs(plt):
            xs = [x / 10.0 for x in range(-50, 50)]
            plt.plot(xs,[normal_pdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
            plt.plot(xs,[normal_pdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')
            plt.plot(xs,[normal_pdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
            plt.plot(xs,[normal_pdf(x,mu=-1)   for x in xs],'-.',label='mu=-1,sigma=1')
            plt.legend()
            plt.show()      

        def normal_cdf(x, mu=0,sigma=1):
            return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2  

        def plot_normal_cdfs(plt):
            xs = [x / 10.0 for x in range(-50, 50)]
            plt.plot(xs,[normal_cdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
            plt.plot(xs,[normal_cdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')
            plt.plot(xs,[normal_cdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
            plt.plot(xs,[normal_cdf(x,mu=-1) for x in xs],'-.',label='mu=-1,sigma=1')
            plt.legend(loc=4) # bottom right
            plt.show()

        def inverse_normal_cdf(p, mu=0, sigma=1, tolerance=0.00001):
            if mu != 0 or sigma != 1:
                return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)
    
            low_z, low_p = -10.0, 0            # normal_cdf(-10) is (very close to) 0
            hi_z,  hi_p  =  10.0, 1            # normal_cdf(10)  is (very close to) 1
            while hi_z - low_z > tolerance:
                mid_z = (low_z + hi_z) / 2     # consider the midpoint
                mid_p = normal_cdf(mid_z)      # and the cdf's value there
                if mid_p < p:
            # midpoint is still too low, search above it
                    low_z, low_p = mid_z, mid_p
                elif mid_p > p:
            # midpoint is still too high, search below it
                    hi_z, hi_p = mid_z, mid_p
                else:
                    break

                return mid_z

        def bernoulli_trial(p):
            return 1 if random.random() < p else 0

        def binomial(p, n):
            return sum(bernoulli_trial(p) for _ in range(n))

        def make_hist(p, n, num_points):
    
            data = [binomial(p, n) for _ in range(num_points)]
    
    # use a bar chart to show the actual binomial samples
            histogram = Counter(data)
            plt.bar([x - 0.4 for x in histogram.keys()],
                     [v / num_points for v in histogram.values()],
                     0.8,
                     color='0.75')
    
            mu = p * n
            sigma = math.sqrt(n * p * (1 - p))

    # use a line chart to show the normal approximation
            xs = range(min(data), max(data) + 1)
            ys = [normal_cdf(i + 0.5, mu, sigma) - normal_cdf(i - 0.5, mu, sigma) 
                  for i in xs]
            plt.plot(xs,ys)
            plt.show()



        if __name__ == "__main__":

    #
    # CONDITIONAL PROBABILITY
    #

            both_girls = 0
            older_girl = 0
            either_girl = 0

            random.seed(0)
            for _ in range(10000):
                younger = random_kid()
                older = random_kid()
                if older == "girl":
                    older_girl += 1
                if older == "girl" and younger == "girl":
                    both_girls += 1
                if older == "girl" or younger == "girl":
                    either_girl += 1

            print ("P(both | older):", both_girls / older_girl)      # 0.514 ~ 1/2
            print ("P(both | either): ", both_girls / either_girl)  
   
e=Button(root,text="GAME MOST PLAYED BOYS,GIRLS OR BOTH",command= ana,padx=30,pady=15)
e.pack(side=LEFT)


root.title("games Reviews")
root.mainloop()