[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/release/python-390/) 
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/license/mit) 
![GitHub version](https://img.shields.io/github/v/release/lgiesen/forest_height?color=green&include_prereleases)

# <img height="60px" src="https://raw.githubusercontent.com/lgiesen/forest_height/main/logo.png" style="margin-right:5px;"/>Title Case
# Data Analytics II Case Study
### Objective
The task of this case study is to determine and approximate the height over forests. This regression is realized with a Machine Learning Regressor and a Convolutional Neural Network (CNN).

### Context
The Federal Ministry for the Environment in Germany is seeking to find the height of every forest in Germany. There are satellite images one can take advantage of to determine the forest height. This regression task requires labels, which are very sparse. In this case study, various approaches are implemented to overcome challenges such as these to determine the forest height in Germany with satellite images. 

### File Structure
- assets: all visualizations from the models
- data: all saved feature combinations as a data basis (only color channels, NDVI, other VIs and all)
- models: ML models
- notebooks: inspect data, train models etc.
- src: python scripts used by notebooks

> NotebookFinal.ipynb: central notebook for submission
> Worth taking a look at: notebooks/compare_models.py to understand how models performed against one another.


![poster](https://github.com/lgiesen/forest_height/blob/main/group-6-poster.png)

### Data
- Training set for training the model
- Public test set for testing the model
- Hidden test set for evaluating the model to determine grades
![data](https://user-images.githubusercontent.com/48908722/236777192-b88a25e5-b151-4998-a33c-3137bb290294.png)
(Source: Lecture slides from [Prof. Gieseke](https://www.linkedin.com/in/fabian-gieseke/))
