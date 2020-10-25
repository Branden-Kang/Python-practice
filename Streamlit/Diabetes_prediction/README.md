# Diabetes Prediction App
Streamlit Web App to predict the onset of diabetes based on diagnostic measures. <br>
Thanks to [Reference](https://medium.com/towards-artificial-intelligence/how-i-build-machine-learning-apps-in-hours-a1b1eaa642ed)

## Data

The data is from the [NIDDK](https://www.niddk.nih.gov/) and we can also get from [Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database)

## Run on Docker
Alternatively you can build the Docker container and access the application at `localhost:8051` on your browser.
```bash
docker build --tag app:1.0 .
docker run --publish 8051:8051 -it app:1.0
```




