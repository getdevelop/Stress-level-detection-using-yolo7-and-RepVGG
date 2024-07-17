
# import argparse
# import io
from PIL import Image
import datetime
import torch#deeplearning lib i used repvgg
import cv2#used to open camera and read imgs from opencv
import numpy as np
import tensorflow as tf#uses simple cnn arch and its deep learning 
from flask import Flask, render_template, request, session,redirect, send_file, url_for, Response, flash
from werkzeug.utils import secure_filename, send_from_directory
import os
import subprocess
from subprocess import Popen
from werkzeug.utils import secure_filename
import os
from emotion import detect_emotion, init
import torch.backends.cudnn as cudnn
from repvgg import create_RepVGG_A0 as create
from flask_wtf import FlaskForm
from wtforms import StringField,SubmitField,IntegerField,PasswordField,EmailField
from wtforms.validators import DataRequired,Email,ValidationError
import bcrypt
from flask_mysqldb import   MySQL
# from werkzeug.security import generate_password_hash, check_password_hash  
from keras.models import load_model
import pymysql 


# registration code and login code starts here
class RegisterForm(FlaskForm):
    name=StringField('Name',validators=[DataRequired()])    
    email=StringField("Email", validators=[DataRequired(),Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit= SubmitField("Register")

class LoginForm(FlaskForm):   
    email=StringField("Email", validators=[DataRequired(),Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit= SubmitField("Login")

class AdminLoginForm(FlaskForm):   
    name=StringField('Name',validators=[DataRequired()])  
    password = PasswordField("Password", validators=[DataRequired()])
    submit= SubmitField("Admin")








app = Flask(__name__)
app.config['SECRET_KEY'] = "hiii"
app.config["UPLOAD_FOLDER"] = 'static/uploads'  # Assuming a static/uploads folder exists
app.config["MYSQL_HOST"] = os.environ.get('MYSQL_HOST', 'localhost')
app.config["MYSQL_USER"] = os.environ.get('MYSQL_USER', 'root')
app.config["MYSQL_PASSWORD"] = os.environ.get('MYSQL_PASSWORD', '')
app.config["MYSQL_DB"] = os.environ.get('MYSQL_DB', 'mydatabase')

mysql=MySQL(app)

def connect_to_database():
    """Connects to the MySQL database."""
    try:
        connection = pymysql.connect(
            host=app.config["MYSQL_HOST"],
            user=app.config["MYSQL_USER"],
            password=app.config["MYSQL_PASSWORD"],
            database=app.config["MYSQL_DB"]
        )
        return connection
    except pymysql.Error as err:
        print("Error connecting to database:", err)
        return None
        
def prediction_image(imagepath):
    model=load_model('model.h5') #trained model of tensoflow 
    
    faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    labels_dict={0:'Angry',1:'Disgust', 2:'Fear', 3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}   
    frame=cv2.imread(imagepath)
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#will convert the pic into grey image
    faces= faceDetect.detectMultiScale(gray, 1.3, 3)
    for x,y,w,h in faces:
        sub_face_img=gray[y:y+h, x:x+w]
        resized=cv2.resize(sub_face_img,(48,48))#resize to
        normalize=resized/255.0
        reshaped=np.reshape(normalize, (1, 48, 48, 1))
        result=model.predict(reshaped)
        label=np.argmax(result, axis=1)[0]
    return labels_dict[label]
    

@app.route('/', methods=['POST'])
def upload_image():
    # labels_dict={0:'Angry',1:'Disgust', 2:'Fear', 3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}
    
    connection = connect_to_database()  

    if connection is None:
        flash('Error connecting to database. Please try again later.')
        return redirect(request.url)

    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        label=prediction_image(filepath)  #functi
        predictions = label
        # print(predictions)
        sql = """INSERT INTO images (filename, predictions) VALUES (%s, %s)"""
        cursor = connection.cursor()
        try:
            cursor.execute(sql, (filename, str(predictions))) 
            connection.commit()
            flash('Image uploaded and saved successfully!')
        except pymysql.Error as err:
            connection.rollback()  
            print("Error saving data to database:", err)
            flash('Error saving data to database. Please try again.')
        finally:
            cursor.close()
            connection.close() 

        return render_template('index.html', filename=filename, predictions=predictions)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/about')
def about():
    return render_template('about.html'); 

@app.route('/display/<filename>')
def display_image(filename):
    """Displays the uploaded image."""
    return redirect(url_for('static', filename='static/uploads/' + filename), code=301)


app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Set upload folder path

model = create(deploy=True)
emotions = ("anger","contempt","disgust","fear","happy","neutral","sad","surprise")

# @app.route("/webcam_feed")
# def webcam_feed():
#     #source = 0
#     cap = cv2.VideoCapture(0)
#     return render_template('index.html')

def init(device):
    global dev
    dev = device
    model.to(device)
    model.load_state_dict(torch.load("weights/repvgg.pth"))
    cudnn.benchmark = True
    model.eval()


@app.route('/open_camera')
def open_camera_page():
    return render_template('open_camera.html')

@app.route('/open_camera_action', methods=['POST','GET'])
def open_camera():
    subprocess.Popen(['python', 'main.py'])
    return 'Camera opened successfully'

@app.route('/close_camera')
def  close_camera():
    open_camera_page().release()
    cv2.destroyAllWindows()
    print("Camera closed successfully!")

def allowed_file(filename):
    """Checks if the uploaded file is a supported image format."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ['png', 'jpg', 'jpeg', 'gif']

@app.route('/')
def home():
    """Renders the home page template."""
    return render_template('index.html')



@app.route('/register', methods=['GET', 'POST'])
def register():
    form=RegisterForm()
    if form.validate_on_submit():
        name=form.name.data
        email=form.email.data
        password=form.password.data
        hashed_password=bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        # hashed_password = generate_password_hash(password)
        cursor=mysql.connection.cursor()
        cursor.execute("INSERT INTO users(name,email,password) VALUES(%s,%s,%s)",(name,email,hashed_password))
        mysql.connection.commit()
        cursor.close()


        return redirect(url_for('login'))
    return render_template('register.html',form=form)



@app.route('/dashboard')
def dashboard():
    if 'user_id' in session:
        user_id = session['user_id']

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users where id=%s",(user_id,))
        user = cursor.fetchone()
        cursor.close()

        if user:
            return render_template('dashboard.html',user=user)
            
    return redirect(url_for('login'))
import datetime






@app.route('/login', methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cursor.fetchone()

        if user and bcrypt.checkpw(password.encode('utf-8'), user[3].encode('utf-8')):
            # Password check with hash
            session['user_id'] = user[0]
            # Insert a new row into the 'history' table for the login event
            login_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute("INSERT INTO history (id, time) VALUES (%s, %s)", (user[0], login_datetime))
            mysql.connection.commit()  # Commit the changes to the database
            cursor.close()
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid email or password!", 'danger')

    return render_template('login.html', form=form)



@app.route('/data')
def data():
    return render_template('data.html')




@app.route('/admin', methods=['GET', 'POST'])
def admin():
    form = AdminLoginForm()
    if form.validate_on_submit():
        name = form.name.data
        password = form.password.data
        
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM admin WHERE name=%s", (name,))
        user = cursor.fetchone()
        
        if user and password:
            session['user_id'] = user[0]
            
            # Fetch user data
            cursor.execute("SELECT * FROM users")
            users_data = cursor.fetchall()
            
            # Fetch login history
            cursor.execute("SELECT id ,time FROM history")
            login_history = cursor.fetchall()
            
            cursor.close()
            
            # Pass user data and login history to the template
            return render_template('data.html', users_data=users_data, login_history=login_history)
        else:
            flash("Invalid username or password!", 'danger')
            return redirect(url_for('admin'))

    return render_template('admin.html', form=form)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash("You have been logged out successfully.")
    return redirect(url_for('login'))



@app.route('/image-data')
def image_data():
    # Check if the user is authenticated as admin
    if 'user_id' not in session:
        return redirect(url_for('admin'))  # Redirect to the admin login page if not logged in
    
    # Fetch image data from the database
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT id, filename, predictions FROM images")
    images = cursor.fetchall()
    cursor.close()
    
    return render_template('image_data.html', images=images)



import matplotlib.pyplot as plt

import io
import base64


# Function to retrieve image predictions from the database
def get_image_predictions():
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT predictions, COUNT(*) as count FROM images GROUP BY precitions") # Assuming 'prediction' is the column name for predictions
    prediction = cursor.fetchall()
    cursor.close()

    return prediction



@app.route('/plot')
def plot():
    # Fetch emotion data from the database
    cur = mysql.connection.cursor()
    cur.execute("SELECT predictions, COUNT(*) as count FROM images GROUP BY predictions")
    data = cur.fetchall()
    cur.close()

    # Process the data for Chart.js
    total_count = sum(row['count'] for row in data)
    labels = []
    percentages = []
    for row in data:
        labels.append(row['predictions'])  # Assuming 'predictions' is the column name
        percentages.append((row['count'] / total_count) * 100)

    return render_template('plot.html', labels=labels, percentages=percentages)





        





if __name__ == '__main__':
    app.secret_key = 'your_secret_key'     
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init(device)
    app.run(debug=True)

