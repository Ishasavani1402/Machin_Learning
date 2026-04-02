from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score

def linear_regression(x_train , y_train):
    model = LinearRegression()
    model.fit(x_train , y_train)
    return model

def decision_tree_regression(x_train , y_train):
    model = DecisionTreeRegressor()
    model.fit(x_train , y_train)
    return model

def random_forest_regression(x_train , y_train):
    model = RandomForestRegressor()
    model.fit(x_train , y_train)
    return model


def evaluate_model(model , x_test , y_test , modelname: str):
    prediction = model.predict(x_test)

    mae = mean_absolute_error(y_test, prediction)
    rmse = mean_squared_error(y_test, prediction)
    r2 = r2_score(y_test, prediction)

    print(f"{modelname}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}%")

    return {
        'model name' : modelname,
        'MAE' : mae,
        'RMSE' : rmse,
        'R2' : r2
    }



