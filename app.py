from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os

import matplotlib.cm as cm
from model.datamodel import ModelTrainer
from model.datamodel import load_data
import pandas as pd

model = None

def init_model():
    global model
    model = ModelTrainer()
    model.load_model('./model/model.pkl', './model/preprocessor.pkl')

def predict(model, file , type='xlsx'):
    if model is None:
        raise ValueError("Модель не была обучена.")
    
    # Предсказания с новой сырыми данными
    if type == 'csv':
        df_new = pd.read_csv(file)
    elif type == 'xlsx':
        df_new = pd.read_excel(file)
    else:
        raise ValueError("Формат файла не поддерживается. Используйте CSV или Excel.")
    predictions = model.predict_with_preprocessing(df_new)

    return predictions

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx'}

import matplotlib.pyplot as plt
import os

# Папка для хранения графиков
GRAPH_FOLDER = 'static/graphs'
os.makedirs(GRAPH_FOLDER, exist_ok=True)

def generate_pie_chart(df):
    plt.figure(figsize=(8, 6))
    
    # Подсчитываем количество автомобилей по маркам
    brand_counts = df['brand'].value_counts(normalize=True)
    
    # Разделяем на категории: больше или меньше 3%
    other_brands = brand_counts[brand_counts < 0.03]
    main_brands = brand_counts[brand_counts >= 0.03]
    
    # Суммируем все маленькие марки в одну категорию "Other"
    other_sum = other_brands.sum()
    main_brands['Other'] = other_sum
    
    # Построение диаграммы
    main_brands.plot(kind='pie', autopct='%1.1f%%', startangle=90)
    plt.title('Распределение марок автомобилей')
    plt.ylabel('')
    pie_chart_path = os.path.join(GRAPH_FOLDER, 'pie_chart.png')
    plt.savefig(pie_chart_path)
    plt.close()
    return pie_chart_path

def generate_histogram(df):
    plt.figure(figsize=(8, 6))
    df['price'].plot(kind='hist', bins=20, color='skyblue', edgecolor='black')
    plt.title('Распределение цен автомобилей')
    plt.xlabel('Цена')
    plt.ylabel('Частота')
    histogram_path = os.path.join(GRAPH_FOLDER, 'histogram.png')
    plt.savefig(histogram_path)
    plt.close()
    return histogram_path

def generate_fuel_histogram(df, selected_brand=None):
    plt.figure(figsize=(10, 6))

    if selected_brand and selected_brand != 'All':
        # Фильтрация данных для выбранной марки
        df = df[df['brand'] == selected_brand]

    # Суммирование автомобилей по типу топлива
    fuel_counts = df['fuel_type'].value_counts()

    # Генерация цветов для столбцов
    colors = cm.get_cmap('tab10', len(fuel_counts))  # Используем палитру 'tab10'
    bar_colors = [colors(i) for i in range(len(fuel_counts))]

    # Построение гистограммы с цветами
    ax = fuel_counts.plot(kind='bar', color=bar_colors, edgecolor='black')
    plt.title('Распределение типов топлива')
    plt.xlabel('Тип топлива')
    plt.ylabel('Количество автомобилей')

    # Добавление подписей над столбцами
    for i, (fuel_type, count) in enumerate(fuel_counts.items()):
        ax.text(i, count + 0.5, str(count), ha='center', va='bottom', fontsize=10)

    # Сохранение графика
    fuel_histogram_path = os.path.join(GRAPH_FOLDER, 'fuel_histogram.png')
    plt.savefig(fuel_histogram_path)
    plt.close()
    return fuel_histogram_path



def generate_line_chart(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['milage'], df['price'], 'o', markersize=4, alpha=0.5)
    plt.title('Зависимость цены от пробега')
    plt.xlabel('Пробег')
    plt.ylabel('Цена')
    line_chart_path = os.path.join(GRAPH_FOLDER, 'line_chart.png')
    plt.savefig(line_chart_path)
    plt.close()
    return line_chart_path


app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('error.html', message='400: No file part')
    file = request.files['file']
    if file.filename == '':
        return render_template('error.html', message='400: No selected file')
    if not allowed_file(file.filename):
        return render_template('error.html', message='400: Unsupported file type. Use CSV or Excel.')

    # Сохранение файла
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Загружаем данные
    df = pd.read_excel(filepath) if filepath.endswith('.xlsx') else pd.read_csv(filepath)

    # Предсказания
    predictions = predict(model, filepath, type=file.filename.rsplit('.', 1)[1].lower())

    # Убеждаемся, что predictions — это DataFrame
    if isinstance(predictions, pd.DataFrame):
        predictions = predictions[['id', 'price']]  # Оставляем только нужные столбцы
        df = df.merge(predictions, on='id', how='left')  # Добавляем предсказания в основной DataFrame

    # Сохраняем новый файл
    new_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'new_{file.filename}')
    if filepath.endswith('.xlsx'):
        df.to_excel(new_filepath, index=False)
    else:
        df.to_csv(new_filepath, index=False)

    # Передаём путь нового файла на страницу analytics
    return redirect(url_for('analytics', filename=f'new_{file.filename}'))


@app.route('/analytics', methods=['GET', 'POST'])
def analytics():
    filename = request.args.get('filename')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Загружаем данные
    df = pd.read_excel(filepath) if filepath.endswith('.xlsx') else pd.read_csv(filepath)

    # Инициализация фильтров
    selected_brand = None
    min_price = None
    max_price = None

    if request.method == 'POST':
        # Получаем данные фильтров из формы
        selected_brand = request.form.get('brand')
        min_price = request.form.get('min_price', type=float)
        max_price = request.form.get('max_price', type=float)

        # Применяем фильтры
        if selected_brand and selected_brand != 'All':
            df = df[df['brand'] == selected_brand]
        if min_price:
            df = df[df['price'] >= min_price]
        if max_price:
            df = df[df['price'] <= max_price]

    # Генерация графиков
    pie_chart = generate_pie_chart(df)
    histogram = generate_histogram(df)
    fuel_histogram = generate_fuel_histogram(df, selected_brand)  # Передаем выбранную марку
    line_chart = generate_line_chart(df)

    # Список доступных марок для фильтра
    brands = ['All'] + sorted(df['brand'].unique())

    return render_template('analytics.html',
                           filename=filename,
                           pie_chart=pie_chart,
                           histogram=histogram,
                           fuel_histogram=fuel_histogram,
                           line_chart=line_chart,
                           brands=brands,
                           selected_brand=selected_brand)


@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html',message='404: Page not found')

if __name__ == '__main__':
    init_model()
    app.run(debug=True)
