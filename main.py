from flask import Flask
from flask_restx import Resource, Api, fields
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import numpy as np


app = Flask(__name__)
api = Api(app)

# Используем данные из sklearn
data = load_iris()
X, y = data.data, data.target

# Доступные модели и их гиперпараметры
model_classes = {'Random Forest': ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"],
                 'Logistic Regression': ["C"],
                 'KNN': ["n_neighbors"]}
# тут храним модельки после обучения
trained_models = {}

# данные для разных методов
a_model = api.model('Model', {'model': fields.String(example="Random Forest"),
                              "n_estimators": fields.Integer(example=100), 'max_depth': fields.Integer(example=2),
                              "min_samples_split": fields.Integer(example=2), "min_samples_leaf": fields.Integer(example=2),
                              "C": fields.Float(example=0.1),
                              "n_neighbors": fields.Integer(example=5)})
a_predict = api.model('Predict', {'model': fields.String(example="Random Forest"),
                                  "input_data": fields.List(fields.Float, example=[1.0, 1.0, 1.0, 1.0])})
a_remove = api.model('Remove', {'model': fields.String(example="Random Forest")})


@api.route('/available_models/')
class Models(Resource):
    def get(self):
        """
        Возвращает список доступных для обучения классов моделей
        """
        message = "You can choose of one three classes of models: "+", ".join(model_classes) + '. '
        message += "You can train one of models again, just make PUT request with the name of already trained model"
        return message

    @api.expect(a_model)
    def put(self):
        """
        Обучает модель с возможностью настройки гиперпараметров. здесь же заново обучать модели
        
        model - название модели, одно из Random Forest, Logistic Regression, KNN
        *params - под названием конкретного гиперпараметра модели подаются соответствующие значения параметров
        """
        model_class = api.payload['model']
        hyper_params = {}
        for model in model_classes:
            if model_class == model:
                for param in model_classes[model]:
                    if api.payload[param] <= 0:
                        message = "All hyperparameters should be positive integers"
                        message += " (except parameter C of logistic regression which should be non-negative float number)"
                        return message
                    else:
                        hyper_params[param] = api.payload[param]
        
        if model_class == 'Random Forest':
            global forest
            forest = RandomForestClassifier(**hyper_params)
            forest.fit(X, y)
            trained_models[model_class] = forest
        elif model_class == 'Logistic Regression':
            global logic
            logic = LogisticRegression(**hyper_params)
            logic.fit(X, y)
            trained_models[model_class] = logic
        elif model_class == 'KNN':
            global knn
            knn = KNeighborsClassifier(**hyper_params)
            knn.fit(X, y)
            trained_models[model_class] = knn
        else:
            message = "This class of models is unavailable. Please, choose one of: "+", ".join(model_classes)
            return message
        
        return f"Model {model_class} was trained"

    @api.expect(a_predict)
    def post(self):
        """
        Возвращает предсказание конкретной модели
        
        model - название обученной модели
        input_data - входные данные соответствующего формата
        """
        model_class = api.payload['model']
        X_test = np.array(api.payload['input_data']).reshape(1, -1)
        if model_class in trained_models.keys():
            y_pred = trained_models[model_class].predict(X_test)
            return f"prediction: {y_pred}"
        else:
            return 'This model is not trained'

    @api.expect(a_remove)
    def delete(self):
        """
        Удаляет уже обученные модели

        model - название обученной модели, которую надо удалить
        """
        model_class = api.payload['model']
        if model_class in trained_models.keys():
            del trained_models[model_class]
            return f"Model {model_class} removed"
        else:
            return 'This model is not trained'


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
