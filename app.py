from flask import Flask, render_template, request, redirect, url_for
import os


from model.datamodel import ModelTrainer
from model.datamodel import load_data
import pandas as pd

model = None

def init_model():
    global model
    model = ModelTrainer()
    model.load_model('./model/model.joblib', './model/preprocessor.joblib')

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
        return render_template('error.html',message='400: No file part')
    file = request.files['file']
    if file.filename == '':
        return render_template('error.html',message='400: No selected file')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

    if not allowed_file(file.filename):
        return render_template('error.html',message='400: Unsupported file type. Use CSV or Excel.')

    return redirect(url_for('analytics', filename=file.filename, file=file))

@app.route('/analytics')
def analytics():
    filename = request.args.get('filename')
    # file = request.args.get('file')

    if filename:
        if filename.endswith('.csv'):
            return render_template('analytics.html', filename='CSV file: ' + filename)
        elif filename.endswith('.xlsx'):
            return render_template('analytics.html', filename='XLSX file: ' + filename)
        else:
            return render_template('analytics.html', filename='Unknown file type')

    return render_template('analytics.html', filename=filename)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html',message='404: Page not found')

if __name__ == '__main__':
    app.run(debug=True)
