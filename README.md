# Real-Estate [Work In Progress]

## Topic Introduction
Predict Austin, TX and Raleigh, NC House Sale Prices Using A Machine Learning Model\. Over time, I am interested in commercial real estate as I acquired more datastat\.

### Purpose & Questions
Interested in real estate markets Austin, TX and Raleigh, NC as they hot real estate markets\. These market have been on a positive incline over these years, and I would like to discover new insights and build a machine learning model to predict the future sales prices per specific features as provided from the dataset\. Which features and variables have the most influence over the sales price movements\. After a preliminary analysis, I'd like to see how other socioeconomic factors impact the sales prices as well\.

### Description of the source data 
1) The Austin, TX data set selected for review and analysis originates from a project at Northwestern University. The csv file was discovered and pulled from [kaggle](https://www.kaggle.com/ericpierce/austinhousingprices)\.

The initial data set from kaggle span from years 2018 to January 2021 and has 15,171 rows and 47 columns of the sales data from Austin, TX\.

2) The Raleigh, NC dataset has not been procured yet.




## EDA and Cleaning Data Set
1. **FINDING**: Single Family(94%), Condo(3%) and Townhouse(1%) make up most of data. **ACTION**: Removed all other home type\.
2. **FINDING**: Possible outliers were lastestPrice, lotSizeSqFt, livingAreaSqFt and numOfBathrooms.  **ACTION**: Corrected typo(s), inputted typical numbers, corrected data based on homeType and latestPrice and used IQR method (used standard 1.5)\.
3. **FINDING**: Bool type data. **ACTION**: Changed to binary data as 0/1\.

### Result
The final data set, saved as 'autinHousingData_cleaned_citynamechanged.csv' has 12,933 rows and 50 columns with 0 missing value\.



## Database
The two datasets that were created during EDA and Cleaning Data set were split based on columns/features that were physically related to the house itself, vs external factors\. 
Looking at the dataset there were several factors that realted to the neighborhood itself, including number of schools (elem, mid, high) as well as school distance and rating\.
It was determined that this information would be split off from the physical characteristics of the house itself (like parking spaces, number of bedrooms, square footage)\.

Based off this, the first table of the dataset being worked included the following columns: zpid, city, streetAddress, zipcode, description, latitude, longitude, propertyTaxRate, garageSpaces, hasAssociation, hasCooling, hasGarage, hasHeating, hasSpa, hasView, homeType, parkingSpaces, yearBuilt, latestPrice, numPriceChanges, latest_saledate, latest_salemonth, latest_saleyear, latestPriceSource, numOfPhotos, accessibility, numOfAppliances, numOfParkingFeatures, patioporch, security, waterfront, windowfeatures, community, lotSizesqFt, livingAreaSqFt, numOfBathrooms, numOfBedrooms, numOfStories, homeImage, zip_rank, median_zip, pr_sqft\. 

The second table had the following columns zpid, numOfPrimarySchools, numOfElementarySchools, numOfMiddleSchools, numOfHighSchools, avgSchoolDistance, avgSchoolRating, avgSchoolSize, MedianStudentsPerTeacher\. Zpid was used as the primary key for both table 1 (named House_data) and table2 (named House_data2)\. 



# Machine Learning with Sklearn
A linear regression model was utilized to predict Austin house prices. See files "Austin-Sklearn-2nd-Demo.ipynb" and "Austin-Sklearn-Demo_chk.ipynb" for further details.

## Correlation and VIF Analysis
With over 40 input features related to each Austin home, a correlation analysis was conducted to help determine the importance of each feature to home prices. In particular, I took out 3 features in the analysis “median_zip, zip_rank, and pr_sqft“ since they are solely meant for the data cleaning process of lurking outlier data within the raw dataset. Looking into the inputs' correlation coefficients, features that fall below 0.5 signify weak to moderate correlation.  In addition to correlation analysis, additional VIF feature calculations were run to not only reduce collinearity among the features but also combat overfitting in the model by removing highly colinear features.

## Multilinear Regression
With the features and output defined, a Sklearn's linear regression function was used. The dataset was split into two for training and testing. p-values were calculated to analyze the model's performance. 

