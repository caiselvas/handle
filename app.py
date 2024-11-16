from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.secret_key = 'tu_clave_secreta'  # Necesaria para mostrar mensajes flash

# Configuración del directorio de carga
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Extensiones permitidas para las imágenes
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    """Verifica si el archivo tiene una extensión permitida."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Procesar las opciones seleccionadas
        feature1 = request.form.get('feature1')
        # Procesar otras características según sea necesario

        # Verificar si se ha subido un archivo
        if 'image' not in request.files:
            flash('No se ha seleccionado ningún archivo.')
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash('No se ha seleccionado ningún archivo.')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Aquí puedes manejar la imagen subida según tus necesidades
            # Por ejemplo, mostrarla en la página o almacenarla para uso futuro

            flash('Archivo subido exitosamente.')
            return render_template('home.html', image_url=filepath)
        else:
            flash('Tipo de archivo no permitido.')
            return redirect(request.url)
    return render_template('home.html')

if __name__ == '__main__':
    # Crear el directorio de carga si no existe
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
