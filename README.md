# WWU: Data Analytics II Case Study
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

NotebookFinal.ipynb: central notebook for submission
Worth taking a look at: notebooks/compare_models.py to understand how models performed against one another.

<hr/>
### Data
- Training set for training the model
- Public test set for testing the model
- Hidden test set for evaluating the model to determine grades
![data](https://user-images.githubusercontent.com/48908722/236777192-b88a25e5-b151-4998-a33c-3137bb290294.png)
(Source: Lecture slides from [Prof. Gieseke](https://www.linkedin.com/in/fabian-gieseke/))

### Link to completed masks
https://uni-muenster.sciebo.de/s/3hsSPnGCSYzz2ur
