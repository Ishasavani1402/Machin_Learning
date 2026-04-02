import joblib
from pathlib import Path
from data_preprocessing import load_vendor_data , prepare_feature , split_data
from model_evaluation import linear_regression , decision_tree_regression , random_forest_regression , evaluate_model

def main():
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)

    # load data
    df = load_vendor_data()

    # prepare and split data
    x , y = prepare_feature(df)
    x_train , x_test , y_train , y_test = split_data(x , y)

    # train models
    linear = linear_regression(x_train , y_train)
    decision_tree = decision_tree_regression(x_train , y_train)
    random_forest = random_forest_regression(x_train , y_train) 

    # evaluate models
    results= []
    results.append(evaluate_model(linear , x_test , y_test , "Linear Regression"))
    results.append(evaluate_model(decision_tree , x_test , y_test , "Decision Tree Regressor"))
    results.append(evaluate_model(random_forest , x_test , y_test , "Random Forest"))

    # select best model
    best_model = min(results , key=lambda x : x['MAE'])
    best_model_name = best_model['model name']

    best_model_object = {
        'Linear Regression' : linear,
        'Decision Tree Regressor' : decision_tree,
        'Random Forest' : random_forest
    }[best_model_name]

    # save best model
    model_path = model_dir / "predict_freight_model.pkl"
    joblib.dump(best_model_object , model_path)

    print(f"Best model: {best_model_name} saved to {model_path}")
    print(f"Model path: {model_path}")


if __name__ == "__main__":
    main()