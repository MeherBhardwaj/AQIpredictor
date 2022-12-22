from flask import Flask, render_template, request, flash
import pandas as pd
import numpy as np
app = Flask(__name__)
app.secret_key = "secret key"
df1 = pd.read_csv("Workable_AQI.csv")
df1_np = df1.to_numpy()
from numpy.linalg import inv


def get_best_model(X, y):
  (n, p) = X.shape
  pf = p + 1

  new_X = np.ones(shape=(n, pf))
  new_X[:, 1:] = X

  return np.dot(np.dot(inv(np.dot(new_X.T, new_X)), new_X.T), y)


def get_predictions(model, X):
   (n, p) = X.shape
   pf = p + 1

   new_X = np.ones(shape=(n, pf))
   new_X[:, 1:] = X

   return np.dot(new_X, model)


@app.route("/output", methods =["GET", "POST"])
def hello():
    if request.method == "POST":
        pm10 = request.form.get("PM10")
        pm2pt5 = request.form.get("PM2.5")
        no = request.form.get("NO")
        no2 = request.form.get("NO2")
        nox = request.form.get("NOx")
        nh3 = request.form.get("NH3")
        co = request.form.get("CO")
        so2 = request.form.get("SO2")
        o3 = request.form.get("O3")
        benzene = request.form.get("Benzene")
        toluene = request.form.get("Toluene")
        xylene = request.form.get("Xylene")
        if pm10 < '0' or pm2pt5 < '0' or no < '0' or no2 < '0' or nox < '0' or nh3 < '0' or co < '0' or so2 < '0' or o3 < '0' or benzene < '0' or toluene < '0' or xylene < '0':
            message = 'Please enter positive concentrations only'
        else:
            X_train = df1_np[:, :12]
            y_train = df1_np[:, -1]
            best_model = get_best_model(X_train, y_train)
            x1 = [pm10, pm2pt5, no, no2, nox, nh3, co, so2, o3, benzene, toluene, xylene]
            X = np.matrix(x1)
            prediction = get_predictions(best_model, X)
            if prediction <= 50:
                output = "Good"
            elif 100 >= prediction > 50:
                output = "Moderate"
            elif 150 >= prediction > 100:
                output = "Unhealthy for Sensitive people"
            elif 200 >= prediction > 150:
                output = "Unhealthy"
            elif 300 >= prediction > 200:
                output = "Very Unhealthy"
            else:
                output = "Hazardous"
            message = "Predicted AQI value is " + str(prediction) + "\nThe Air Quality is " + output
        flash(message)
        return render_template("output.html")
    return render_template("output.html")


@app.route("/")
def index():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)














#app = Flask(__name__)


#@app.route("/")
#def index():
 #   return render_template('index.html')


#if __name__ == '__main__':
 #   app.run(debug=True)


#@app.route("/accept", method=['POST'])
#def move_forward():
 #   forward_message = "Moving Forward..."
  #  return render_template('index.html', forward_message=forward_message)


#if __name__ == '__main__':
 #   app.run(debug=True)
