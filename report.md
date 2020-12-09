# Report Project Mines ML 2020 -  Apply DL model for Traffic Volume prediction

Author: Hiba Najjar

## Overview
 To fill later

## Data exploration

([Link](https://www.kaggle.com/vinayshanbhag/radar-traffic-data) to the Kaggle Dataset of *Radar Traffic Data*)

We'll be working on a dataframe of **12** columns and **4.6M** records.

-  3 location columns: location name, longitude and latitude, with 18 different locations in total;
- 7 date columns: Year, Month, Day, Day of the Week, Hour, Minute and Time. No missing value was detected in these columns.
 <img src='/Figures/Histogramms.png'>
 
- a column for the directions, (SB, NB, EB, WB), with 5.6% of missing values: 

|    SB   |    NB   |   EB   |   WB   |  NaN  |
|:-------:|:-------:|:------:|:------:|:------:|
| 2 035 591 | 1 714 696 | 328 580 | 262 926 | 262 068 |

- and finally the Volume column, or number of vehicles detected by the sensor, in the last 15 minutes:

|   count   |    mean   |    std    | min |  25% |  50% |  75%  |  max  |
|:---------:|:---------:|:---------:|:---:|:----:|:----:|:-----:|:-----:|
| 4 603 861 | 71.17 | 63.70 | 0 | 13 | 56 | 115 | 255 |



## Can we use PCA to fill missing Direction values ?
Inspired by this [article](https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60), we will use PCA to reduce dimension of our data and draw scatter plots on the two principal components of our records, using the `Direction` variable as a color factor :

<img src="/Figures/PCA.png">

The results show that this approach may not be the best to impute the missing values. Other methods can solve this problematic, but for the rest of this projet we will simply drop records with missing values.


## Predictors Selection

Not all the columns we have in our original dataset are necessarly useful for our Volume Prediction task. Therefore, we will first look into correlation between variables, and then examine boxplot  of Volume values grouped by all categorical variables:  

<img src="/Figures/corr_matrix.png">

<img src="/Figures/boxplots.png">

Looking at **Volume-Date** boxplots, it would be reasonable to only keep `Hour`, `Month` and `Day of Week` predictors, and for the **Location** variables, we'll use `location_name` only. We will also update the value of `Minute` using the the `Time Bin` variable before we get ride off it.

Therefore, our data will be composed of the following categorical variables:
|       Varibale       | Location Name | Direction | Month | Day of the Week | Hour | Minute |
|:--------------------:|:-------------:|:---------:|:-----:|:---------------:|:----:|:------:|
| Number of categories |       16      |     4     |   12  |        7        |  24  |    4   |

Categorical variables will be divided into batches under an approriate representation ( stack of one-hot vector per categorical variable) for the training and test steps.














# Template

Authors: John Doo and Marie Curie

## Recommendations

- You may write this report in French

## How to write a report in markdown

- The syntax is very simple: it's just raw texts with special formatting for titles, images, equations, etc.
- Get a look [here](https://www.markdownguide.org/cheat-sheet/) for a quick guide about the syntax
- Including an image like this: <img src="dephell.png" width="50%"/> (*attribution: XKCD.com*)
- Including a table is a bit annoying, but most good editors have plugins to easily edit markdown tables:

| Left-aligned | Center-aligned | Right-aligned |
| :---         |     :---:      |          ---: |
| git status   | git status     | git status    |
| git diff     | git diff       | git diff      |

- For a quotation with reference: "...the **go to** statement should be abolished..." [[1]](#1).

- If you want a PDF version of your report, you may convert it with pandoc:

```
   pandoc -s report.md -o report.pdf
```

or, if you use the img tag, you have to first convert into pure html:

```
pandoc -t html report.md | pandoc -f html -o report.pdf
```

## References
<a id="1">[1]</a> 
Dijkstra, E. W. (1968). 
Go to statement considered harmful. 
Communications of the ACM, 11(3), 147-148.

