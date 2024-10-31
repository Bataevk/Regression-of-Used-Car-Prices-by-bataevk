import joblib
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import category_encoders as ce
from sklearn.model_selection import train_test_split  

from icecream import ic

class DataPreprocessor:
    def __init__(self):
        # Инициализация предобработчиков данных
        self.imputer = KNNImputer(n_neighbors=5)
        self.target_encoder = ce.TargetEncoder(cols=['brand', 'model', 'fuel_type', 'transmission', 'ext_col', 'int_col'])
        self.scaler = RobustScaler()
        self.min_max_scaler = MinMaxScaler()
        self.price_scaler = MinMaxScaler()
        self.price_imputer = KNNImputer(n_neighbors=5)
        self.is_fitted = False

    def filter_price(self, df):  
        # Расчёт Q1 и Q3
        Q1 = df['price'].quantile(0.25)  
        Q3 = df['price'].quantile(0.75)  

        # Расчёт IQR
        IQR = Q3 - Q1
        
        # Определение границ
        lower_bound = Q1 - 1.0 * IQR
        upper_bound = Q3 + 1.0 * IQR
        # Фильтрация данных от выбросов
        filtered_data = ic(df[(df_train['price'] >= 5_000) & (df_train['price'] <= 130_000)])

        return filtered_data

    def fit(self, df, current_year=2024):
        # Обучение предобработчиков
        df[['price']] = self.price_imputer.fit_transform(df[['price']])
        df[['price']] = self.price_scaler.fit_transform(df[['price']])
        numeric_cols = ['milage']
        self.imputer.fit(df[numeric_cols])
        self.target_encoder.fit(df[['brand', 'model', 'fuel_type', 'transmission', 'ext_col', 'int_col']], df['price'])
        self.scaler.fit(df[numeric_cols])
        self.min_max_scaler.fit(df['engine'].str.extract(r'(\d+) Cylinder').astype(float))
        self.min_year, self.max_year = df['model_year'].min(), current_year
        self.is_fitted = True

    def transform(self, df, dropna=True, drop_id=True):
        # Преобразование данных после обучения предобработчиков
        if not self.is_fitted:
            raise ValueError("Трансформатор не был обучен. Сначала вызовите 'fit'.")
        numeric_cols = ['milage']
        df[numeric_cols] = self.imputer.transform(df[numeric_cols])
        df[['brand_encoded', 'model_encoded', 'fuel_encoded', 'transmission_encoded', 'ext_col_encoded', 'int_col_encoded']] = self.target_encoder.transform(
            df[['brand', 'model', 'fuel_type', 'transmission', 'ext_col', 'int_col']]
        )
        df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        df['num_cylinders'] = df['engine'].str.extract(r'(\d+) Cylinder').astype(float)
        df['num_cylinders'] = self.min_max_scaler.transform(df[['num_cylinders']])
        df['model_year'] = (df['model_year'] - self.min_year) / (self.max_year - self.min_year)

        if drop_id:
            df.drop(columns=['id'], inplace=True)
        df.drop(columns=['brand', 'model', 'fuel_type', 'transmission', 'ext_col', 'int_col', 'engine', 'accident', 'clean_title'], inplace=True)
        if dropna:
            df.dropna(inplace=True)

        return df

    def transform_price(self, df_price):
        return self.price_scaler.transform(df_price[['price']])
    
    def inverse_transform_price(self, predictions):
        # Возвращаем предсказания обратно в исходный масштаб цен
        return self.price_scaler.inverse_transform(predictions.reshape(-1, 1))

    def save_preprocessor(self, filepath):
        # Сохраняем все объекты предобработки
        preprocessor_objects = {
            'imputer': self.imputer,
            'target_encoder': self.target_encoder,
            'scaler': self.scaler,
            'min_max_scaler': self.min_max_scaler,
            'price_scaler': self.price_scaler,
            'price_imputer': self.price_imputer
        }
        joblib.dump(preprocessor_objects, filepath)

    def load_preprocessor(self, filepath):
        # Загружаем предобработчики
        preprocessor_objects = joblib.load(filepath)
        self.imputer = preprocessor_objects['imputer']
        self.target_encoder = preprocessor_objects['target_encoder']
        self.scaler = preprocessor_objects['scaler']
        self.min_max_scaler = preprocessor_objects['min_max_scaler']
        self.price_scaler = preprocessor_objects['price_scaler']
        self.price_imputer = preprocessor_objects['price_imputer']
        self.is_fitted = True

class ModelTrainer:
    def __init__(self):
        self.model = None
        self.preprocessor = DataPreprocessor()

    def train(self, X_train, y_train):
        # Обучение модели
        self.model = XGBRegressor(n_estimators=100, max_depth=4, random_state=42, learning_rate=0.1)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Модель не была обучена.")
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test, preprocessor):
        # Оценка модели
        predictions = self.predict(X_test)
        y_test_transformed = preprocessor.transform_price(y_test)
        mae = mean_absolute_error(y_test_transformed, predictions)
        mse = mean_squared_error(y_test_transformed, predictions)
        rmse = mean_squared_error(y_test_transformed, predictions, squared=False)
        r2 = r2_score(y_test_transformed, predictions)
        return mae, mse, rmse, r2

    def save_model(self, model_filepath, preprocessor_filepath):
        # Сохраняем модель и предобработчики
        joblib.dump(self.model, model_filepath)
        self.preprocessor.save_preprocessor(preprocessor_filepath)

    def load_model(self, model_filepath, preprocessor_filepath):
        # Загружаем модель и предобработчики
        self.model = joblib.load(model_filepath)
        self.preprocessor.load_preprocessor(preprocessor_filepath)

    def predict_with_preprocessing(self, X_raw, real_price = True):
        # Преобразуем данные и возвращаем предсказания
        X_preprocessed = self.preprocessor.transform(ic(X_raw), False, False)

        X_test = X_preprocessed.drop(columns=['id'])


        X_test.fillna(0, inplace=True)


        submission = X_preprocessed[['id']]


        # Предсказание на final тестовых данных
        f_predictions = self.predict(X_test)

        if real_price:
            f_predictions_price = self.preprocessor.price_scaler.inverse_transform(f_predictions.reshape(-1, 1))

        submission['price'] = f_predictions_price

        return submission

# Функции для загрузки данных
def load_data(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Формат файла не поддерживается. Используйте CSV или Excel.")


if __name__ == "__main__":
    # Загрузка данных
    df_train = load_data('./train.csv')
    df_train, df_test = train_test_split(df_train, test_size=0.1, random_state=42)

    # Загрузка главного модуля
    trainer = ModelTrainer()


    # Фильтрация выбросов для обучения
    df_train = trainer.preprocessor.filter_price(df_train)

    # Предобработка данных
    preprocessor = trainer.preprocessor
    preprocessor.fit(ic(df_train))
    df_train = preprocessor.transform(df_train)
    X_train = df_train.drop(columns=['price'])
    y_train = df_train[['price']]

    # Обучение модели
    trainer.train(ic(X_train), ic(y_train))

    # Оценка модели
    df_test = preprocessor.transform(df_test)
    X_test = df_test.drop(columns=['price'])
    y_test = df_test[['price']]
    mae, mse, rmse, r2 = trainer.evaluate(X_test, y_test, preprocessor)
    print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}")

    # Сохранение модели и предобработчиков
    trainer.save_model('./trained_model.pkl', './preprocessor.pkl')


    # Предсказания с новой сырыми данными
    df_new = load_data('./test.csv')  # это новые данные
    predictions = trainer.predict_with_preprocessing(df_new)

    # Сохранение предсказаний
    predictions.to_csv('./submission.csv', index=False)
