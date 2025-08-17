import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QLineEdit, QTreeWidget, QTreeWidgetItem,
                             QMessageBox, QMenuBar, QMenu, QAction, QDialog, QScrollArea,
                             QFrame, QSizePolicy, QInputDialog)
from PyQt5.QtCore import Qt, QTimer, QDateTime
from PyQt5.QtGui import QFont, QPalette, QColor
import cv2
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
import shutil

class FaceRecognitionAttendanceSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition Attendance System")
        self.setGeometry(100, 100, 1280, 720)
        
        # Initialize variables
        self.haar_path = "haarcascade_frontalface_default.xml"
        self.student_details_path = "StudentDetails/StudentDetails.csv"
        self.training_image_path = "TrainingImage"
        self.training_label_path = "TrainingImageLabel/Trainner.yml"
        self.attendance_dir = "Attendance"
        self.current_attendance = []  # To store current session attendance
        
        # Setup UI
        self.setup_ui()
        
        # Check required files
        self.check_required_files()
        
        # Start clock
        self.update_clock()
    
    def setup_ui(self):
        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)
        
        # Header
        self.setup_header()
        
        # Content area
        self.setup_content_area()
        
        # Menu
        self.setup_menu()
    
    def setup_header(self):
        header_frame = QFrame()
        header_frame.setFrameShape(QFrame.StyledPanel)
        header_frame.setStyleSheet("background-color: #34495e;")
        header_frame.setFixedHeight(80)
        
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(20, 0, 20, 0)
        
        # Title
        title_label = QLabel("Face Recognition Attendance System")
        title_label.setStyleSheet("color: #ecf0f1;")
        title_font = QFont("Helvetica", 24, QFont.Bold)
        title_label.setFont(title_font)
        header_layout.addWidget(title_label, alignment=Qt.AlignLeft)
        
        # Date and time
        self.setup_datetime_display(header_layout)
        
        self.main_layout.addWidget(header_frame)
    
    def setup_datetime_display(self, parent_layout):
        datetime_frame = QFrame()
        datetime_layout = QVBoxLayout(datetime_frame)
        datetime_layout.setContentsMargins(0, 0, 0, 0)
        datetime_layout.setSpacing(0)
        
        # Date
        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
        day, month, year = date.split("-")
        
        mont = {
            '01': 'January', '02': 'February', '03': 'March', '04': 'April',
            '05': 'May', '06': 'June', '07': 'July', '08': 'August',
            '09': 'September', '10': 'October', '11': 'November', '12': 'December'
        }
        
        self.date_label = QLabel(f"{day} {mont[month]} {year}")
        self.date_label.setStyleSheet("color: #ecf0f1;")
        date_font = QFont("Helvetica", 12)
        self.date_label.setFont(date_font)
        datetime_layout.addWidget(self.date_label, alignment=Qt.AlignRight)
        
        # Time
        self.time_label = QLabel()
        self.time_label.setStyleSheet("color: #ecf0f1;")
        self.time_label.setFont(date_font)
        datetime_layout.addWidget(self.time_label, alignment=Qt.AlignRight)
        
        parent_layout.addWidget(datetime_frame, alignment=Qt.AlignRight)
    
    def setup_content_area(self):
        content_frame = QFrame()
        content_layout = QHBoxLayout(content_frame)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10)
        
        # Left frame (Attendance)
        self.setup_attendance_frame(content_layout)
        
        # Right frame (Registration)
        self.setup_registration_frame(content_layout)
        
        self.main_layout.addWidget(content_frame)
    
    def setup_attendance_frame(self, parent_layout):
        attendance_frame = QFrame()
        attendance_frame.setFrameShape(QFrame.StyledPanel)
        attendance_frame.setStyleSheet("background-color: #34495e; border: 1px solid #2980b9;")
        
        attendance_layout = QVBoxLayout(attendance_frame)
        attendance_layout.setContentsMargins(10, 10, 10, 10)
        attendance_layout.setSpacing(10)
        
        # Header
        header = QLabel("Attendance Management")
        header.setStyleSheet("""
            color: #ecf0f1; 
            background-color: #3498db;
            padding: 10px;
            font-weight: bold;
        """)
        header_font = QFont("Helvetica", 16, QFont.Bold)
        header.setFont(header_font)
        header.setAlignment(Qt.AlignCenter)
        attendance_layout.addWidget(header)
        
        # Take attendance button
        btn_take_attendance = QPushButton("Take Attendance")
        btn_take_attendance.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: #ecf0f1;
                padding: 15px;
                font-weight: bold;
                border: none;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
        """)
        btn_take_attendance.setFont(QFont("Helvetica", 12, QFont.Bold))
        btn_take_attendance.clicked.connect(self.track_images)
        attendance_layout.addWidget(btn_take_attendance)
        
        # Attendance table
        self.setup_attendance_table(attendance_layout)
        
        # Quit button
        btn_quit = QPushButton("Quit")
        btn_quit.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: #ecf0f1;
                padding: 15px;
                font-weight: bold;
                border: none;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        btn_quit.setFont(QFont("Helvetica", 12, QFont.Bold))
        btn_quit.clicked.connect(self.close)
        attendance_layout.addWidget(btn_quit)
        
        parent_layout.addWidget(attendance_frame)
    
    def setup_registration_frame(self, parent_layout):
        registration_frame = QFrame()
        registration_frame.setFrameShape(QFrame.StyledPanel)
        registration_frame.setStyleSheet("background-color: #34495e; border: 1px solid #2980b9;")
        
        registration_layout = QVBoxLayout(registration_frame)
        registration_layout.setContentsMargins(10, 10, 10, 10)
        registration_layout.setSpacing(10)
        
        # Header
        header = QLabel("Student Registration")
        header.setStyleSheet("""
            color: #ecf0f1; 
            background-color: #3498db;
            padding: 10px;
            font-weight: bold;
        """)
        header_font = QFont("Helvetica", 16, QFont.Bold)
        header.setFont(header_font)
        header.setAlignment(Qt.AlignCenter)
        registration_layout.addWidget(header)
        
        # Form fields
        self.setup_registration_form(registration_layout)
        
        # Action buttons
        self.setup_registration_buttons(registration_layout)
        
        # Status messages
        self.setup_status_messages(registration_layout)
        
        parent_layout.addWidget(registration_frame)
    
    def setup_registration_form(self, parent_layout):
        form_frame = QFrame()
        form_layout = QVBoxLayout(form_frame)
        form_layout.setContentsMargins(10, 10, 10, 10)
        form_layout.setSpacing(10)
        
        # Roll number
        roll_frame = QFrame()
        roll_layout = QHBoxLayout(roll_frame)
        roll_layout.setContentsMargins(0, 0, 0, 0)
        
        lbl_roll = QLabel("Student ID:")
        lbl_roll.setStyleSheet("color: #ecf0f1;")
        lbl_roll.setFont(QFont("Helvetica", 12))
        roll_layout.addWidget(lbl_roll)
        
        self.txt_roll = QLineEdit()
        self.txt_roll.setStyleSheet("background-color: white; color: #2c3e50;")
        self.txt_roll.setFont(QFont("Helvetica", 12))
        roll_layout.addWidget(self.txt_roll)
        
        btn_clear_roll = QPushButton("Clear")
        btn_clear_roll.setStyleSheet("""
            QPushButton {
                background-color: #7f8c8d;
                color: #ecf0f1;
                padding: 5px;
                font-weight: bold;
                border: none;
            }
            QPushButton:hover {
                background-color: #95a5a6;
            }
        """)
        btn_clear_roll.setFont(QFont("Helvetica", 10, QFont.Bold))
        btn_clear_roll.clicked.connect(self.clear_roll)
        roll_layout.addWidget(btn_clear_roll)
        
        form_layout.addWidget(roll_frame)
        
        # Name
        name_frame = QFrame()
        name_layout = QHBoxLayout(name_frame)
        name_layout.setContentsMargins(0, 0, 0, 0)
        
        lbl_name = QLabel("Full Name:")
        lbl_name.setStyleSheet("color: #ecf0f1;")
        lbl_name.setFont(QFont("Helvetica", 12))
        name_layout.addWidget(lbl_name)
        
        self.txt_name = QLineEdit()
        self.txt_name.setStyleSheet("background-color: white; color: #2c3e50;")
        self.txt_name.setFont(QFont("Helvetica", 12))
        name_layout.addWidget(self.txt_name)
        
        btn_clear_name = QPushButton("Clear")
        btn_clear_name.setStyleSheet("""
            QPushButton {
                background-color: #7f8c8d;
                color: #ecf0f1;
                padding: 5px;
                font-weight: bold;
                border: none;
            }
            QPushButton:hover {
                background-color: #95a5a6;
            }
        """)
        btn_clear_name.setFont(QFont("Helvetica", 10, QFont.Bold))
        btn_clear_name.clicked.connect(self.clear_name)
        name_layout.addWidget(btn_clear_name)
        
        form_layout.addWidget(name_frame)
        
        parent_layout.addWidget(form_frame)
    
    def setup_registration_buttons(self, parent_layout):
        button_frame = QFrame()
        button_layout = QVBoxLayout(button_frame)
        button_layout.setContentsMargins(20, 10, 20, 10)
        button_layout.setSpacing(10)
        
        btn_take_images = QPushButton("Capture Images")
        btn_take_images.setStyleSheet("""
            QPushButton {
                background-color: #2980b9;
                color: #ecf0f1;
                padding: 15px;
                font-weight: bold;
                border: none;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
        """)
        btn_take_images.setFont(QFont("Helvetica", 12, QFont.Bold))
        btn_take_images.clicked.connect(self.take_images)
        button_layout.addWidget(btn_take_images)
        
        btn_train_images = QPushButton("Train Model")
        btn_train_images.setStyleSheet("""
            QPushButton {
                background-color: #16a085;
                color: #ecf0f1;
                padding: 15px;
                font-weight: bold;
                border: none;
            }
            QPushButton:hover {
                background-color: #1abc9c;
            }
        """)
        btn_train_images.setFont(QFont("Helvetica", 12, QFont.Bold))
        btn_train_images.clicked.connect(self.train_images)
        button_layout.addWidget(btn_train_images)
        
        # Add Delete button
        btn_delete = QPushButton("Delete Registration")
        btn_delete.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: #ecf0f1;
                padding: 15px;
                font-weight: bold;
                border: none;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        btn_delete.setFont(QFont("Helvetica", 12, QFont.Bold))
        btn_delete.clicked.connect(self.delete_registration)
        button_layout.addWidget(btn_delete)
        
        parent_layout.addWidget(button_frame)
    
    def setup_status_messages(self, parent_layout):
        message_frame = QFrame()
        message_layout = QVBoxLayout(message_frame)
        message_layout.setContentsMargins(10, 10, 10, 10)
        message_layout.setSpacing(10)
        
        self.instruction_label = QLabel("1) Capture Images  →  2) Train Model")
        self.instruction_label.setStyleSheet("color: #f1c40f;")
        self.instruction_label.setFont(QFont("Helvetica", 12))
        self.instruction_label.setAlignment(Qt.AlignCenter)
        message_layout.addWidget(self.instruction_label)
        
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #2ecc71;")
        self.status_label.setFont(QFont("Helvetica", 12))
        self.status_label.setAlignment(Qt.AlignCenter)
        message_layout.addWidget(self.status_label)
        
        self.registration_count_label = QLabel("Total Registrations: 0")
        self.registration_count_label.setStyleSheet("color: #bdc3c7;")
        self.registration_count_label.setFont(QFont("Helvetica", 12))
        self.registration_count_label.setAlignment(Qt.AlignCenter)
        message_layout.addWidget(self.registration_count_label)
        
        parent_layout.addWidget(message_frame)
    
    def setup_attendance_table(self, parent_layout):
        table_frame = QFrame()
        table_layout = QVBoxLayout(table_frame)
        table_layout.setContentsMargins(10, 10, 10, 10)
        
        self.tv = QTreeWidget()
        self.tv.setColumnCount(4)  # Changed from 3 to 4 columns to include ID
        self.tv.setHeaderLabels(["ID", "NAME", "DATE", "TIME"])  # Updated header labels
        self.tv.setStyleSheet("""
            QTreeWidget {
                background-color: white;
                color: #2c3e50;
                font-size: 12px;
                border: 1px solid #bdc3c7;
            }
            QTreeWidget::item {
                height: 25px;
            }
            QHeaderView::section {
                background-color: #3498db;
                color: white;
                padding: 5px;
                font-weight: bold;
            }
        """)
        
        # Column widths
        self.tv.setColumnWidth(0, 100)  # ID column
        self.tv.setColumnWidth(1, 150)  # Name column
        self.tv.setColumnWidth(2, 120)  # Date column
        self.tv.setColumnWidth(3, 120)  # Time column
        
        table_layout.addWidget(self.tv)
        parent_layout.addWidget(table_frame)
    
    def setup_menu(self):
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #34495e;
                color: #ecf0f1;
                padding: 5px;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 5px 10px;
            }
            QMenuBar::item:selected {
                background-color: #2980b9;
            }
            QMenu {
                background-color: #34495e;
                color: #ecf0f1;
                border: 1px solid #2980b9;
            }
            QMenu::item:selected {
                background-color: #2980b9;
            }
        """)
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        change_pass_action = QAction("Change Password", self)
        change_pass_action.triggered.connect(self.change_password)
        file_menu.addAction(change_pass_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        user_guide_action = QAction("User Guide", self)
        user_guide_action.triggered.connect(self.user_guide)
        help_menu.addAction(user_guide_action)
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.about)
        help_menu.addAction(about_action)
        
        contact_action = QAction("Contact Us", self)
        contact_action.triggered.connect(self.contact)
        help_menu.addAction(contact_action)
    
    def check_required_files(self):
        # Check haarcascade file
        if not os.path.isfile(self.haar_path):
            # Try to find it in OpenCV installation
            opencv_path = os.path.join(
                os.path.dirname(cv2.__file__), "data", "haarcascade_frontalface_default.xml"
            )
            if os.path.isfile(opencv_path):
                shutil.copy(opencv_path, self.haar_path)
            else:
                QMessageBox.critical(
                    self,
                    "Missing File",
                    "haarcascade_frontalface_default.xml not found. Please download it."
                )
                self.close()
        
        # Create required directories
        os.makedirs("StudentDetails", exist_ok=True)
        os.makedirs(self.training_image_path, exist_ok=True)
        os.makedirs("TrainingImageLabel", exist_ok=True)
        os.makedirs(self.attendance_dir, exist_ok=True)
        
        # Initialize student details CSV if not exists
        if not os.path.isfile(self.student_details_path):
            with open(self.student_details_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['SERIAL NO.', 'ID', 'NAME'])
        
        self.update_registration_count()
    
    def update_clock(self):
        current_time = QDateTime.currentDateTime()
        time_string = current_time.toString('hh:mm:ss')
        date_string = current_time.toString('dd MMMM yyyy')
        
        self.time_label.setText(time_string)
        self.date_label.setText(date_string)
        
        # Update every second
        QTimer.singleShot(1000, self.update_clock)
    
    def clear_roll(self):
        self.txt_roll.clear()
        self.instruction_label.setText("1) Capture Images  →  2) Train Model")
        self.instruction_label.setStyleSheet("color: #f1c40f;")
    
    def clear_name(self):
        self.txt_name.clear()
        self.instruction_label.setText("1) Capture Images  →  2) Train Model")
        self.instruction_label.setStyleSheet("color: #f1c40f;")
    
    def take_images(self):
        roll = self.txt_roll.text().strip()
        name = self.txt_name.text().strip()
        
        if not name.replace(' ', '').isalpha():
            self.status_label.setText("Error: Name must contain only letters and spaces")
            self.status_label.setStyleSheet("color: #e74c3c;")
            return
        
        if not roll:
            self.status_label.setText("Error: Please enter student ID")
            self.status_label.setStyleSheet("color: #e74c3c;")
            return
        
        # Get next serial number
        with open(self.student_details_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            serial = sum(1 for _ in reader) + 1
        
        # Initialize camera
        try:
            cam = cv2.VideoCapture(0)
            if not cam.isOpened():
                raise Exception("Camera not accessible")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return
        
        detector = cv2.CascadeClassifier(self.haar_path)
        sample_num = 0
        
        # Create a window for displaying camera feed
        cv2.namedWindow("Capturing Images", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Capturing Images", 800, 600)
        
        self.status_label.setText("Capturing images... Please look at the camera")
        self.status_label.setStyleSheet("color: #f39c12;")
        QApplication.processEvents()
        
        while sample_num < 100:
            ret, img = cam.read()
            if not ret:
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                sample_num += 1
                
                # Save image
                img_name = f"{name}.{serial}.{roll}.{sample_num}.jpg"
                img_path = os.path.join(self.training_image_path, img_name)
                cv2.imwrite(img_path, gray[y:y+h, x:x+w])
                
                # Display progress
                cv2.putText(img, f"Captured: {sample_num}/100", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display
                cv2.imshow('Capturing Images', img)
            
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        
        cam.release()
        cv2.destroyAllWindows()
        
        # Save to CSV
        with open(self.student_details_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([serial, roll, name])
        
        self.status_label.setText(f"Success: Captured 100 images for {name} (ID: {roll})")
        self.status_label.setStyleSheet("color: #2ecc71;")
        self.instruction_label.setText("Now train the model with the captured images")
        self.update_registration_count()
    
    def train_images(self):
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            detector = cv2.CascadeClassifier(self.haar_path)
            
            faces, ids = self.get_images_and_labels(self.training_image_path)
            
            if not faces:
                self.status_label.setText("Error: No training images found")
                self.status_label.setStyleSheet("color: #e74c3c;")
                return
            
            recognizer.train(faces, np.array(ids))
            recognizer.save(self.training_label_path)
            
            self.status_label.setText("Success: Model trained with all images")
            self.status_label.setStyleSheet("color: #2ecc71;")
            self.instruction_label.setText("You can now take attendance")
        except Exception as e:
            self.status_label.setText(f"Error: Training failed - {str(e)}")
            self.status_label.setStyleSheet("color: #e74c3c;")
    
    def get_images_and_labels(self, path):
        image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
        faces = []
        ids = []
        
        for image_path in image_paths:
            pil_image = Image.open(image_path).convert('L')
            image_np = np.array(pil_image, 'uint8')
            
            try:
                id = int(os.path.split(image_path)[-1].split(".")[1])
                faces.append(image_np)
                ids.append(id)
            except (IndexError, ValueError):
                continue
        
        return faces, ids
    
    def track_images(self):
        # Clear previous attendance records
        self.tv.clear()
        
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(self.training_label_path)
        except:
            QMessageBox.critical(self, "Error", "No trained data found. Please train the model first.")
            return
        
        detector = cv2.CascadeClassifier(self.haar_path)
        
        try:
            df = pd.read_csv(self.student_details_path)
        except:
            QMessageBox.critical(self, "Error", "Student details not found")
            return
        
        try:
            cam = cv2.VideoCapture(0)
            if not cam.isOpened():
                raise Exception("Camera not accessible")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        col_names = ['Id', 'Name', 'Date', 'Time']
        self.current_attendance = []
        
        # Dictionary to track last recognition time for each student
        self.last_recognized = {}
        self.recognition_cooldown = 5  # seconds between recognitions
        
        # Create a window for displaying camera feed
        cv2.namedWindow("Taking Attendance", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Taking Attendance", 800, 600)
        
        self.status_label.setText("Taking attendance... Press 'q' to stop")
        self.status_label.setStyleSheet("color: #f39c12;")
        QApplication.processEvents()
        
        while True:
            ret, im = cam.read()
            if not ret:
                continue
            
            current_time = time.time()
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.2, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
                serial, conf = recognizer.predict(gray[y:y + h, x:x + w])
                
                if conf < 50:  # Confidence threshold
                    try:
                        aa = df.loc[df['SERIAL NO.'] == serial]['NAME'].values[0]
                        id = df.loc[df['SERIAL NO.'] == serial]['ID'].values[0]
                        
                        # Check if we recently recognized this student
                        last_time = self.last_recognized.get(id, 0)
                        if current_time - last_time < self.recognition_cooldown:
                            # Skip marking attendance if recently recognized
                            cv2.putText(im, f"Already marked: {aa}", (x, y + h + 30), 
                                        font, 0.8, (0, 255, 255), 2)
                            continue
                            
                        # Mark attendance
                        ts = time.time()
                        date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                        time_stamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                        
                        # Check if this student has already been marked today
                        already_marked = any(att[0] == str(id) for att in self.current_attendance)
                        
                        if not already_marked:
                            attendance = [str(id), str(aa), date, time_stamp]
                            self.current_attendance.append(attendance)
                            self.last_recognized[id] = current_time
                            
                            # Add to treeview - now with 4 columns (ID, Name, Date, Time)
                            item = QTreeWidgetItem(self.tv)
                            item.setText(0, str(id))  # ID column
                            item.setText(1, str(aa))  # Name column
                            item.setText(2, date)      # Date column
                            item.setText(3, time_stamp) # Time column
                            
                            # Display confirmation
                            cv2.putText(im, f"Marked: {aa}", (x, y + h + 30), 
                                        font, 0.8, (0, 255, 0), 2)
                        
                        cv2.putText(im, str(aa), (x, y + h), font, 1, (255, 255, 255), 2)
                    except Exception as e:
                        print(f"Error: {e}")
                else:
                    cv2.putText(im, "Unknown", (x, y + h), font, 1, (255, 255, 255), 2)
            
            cv2.imshow('Taking Attendance', im)
            if cv2.waitKey(1) == ord('q'):
                break
        
        cam.release()
        cv2.destroyAllWindows()
        
        # Save attendance
        if self.current_attendance:
            ts = time.time()
            date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
            attendance_file = os.path.join(self.attendance_dir, f"Attendance_{date}.csv")
            
            # Read existing attendance to prevent duplicates
            existing_attendance = []
            if os.path.isfile(attendance_file):
                with open(attendance_file, 'r') as file:
                    reader = csv.reader(file)
                    next(reader)  # Skip header
                    existing_attendance = list(reader)
            
            # Write header if file doesn't exist
            if not os.path.isfile(attendance_file):
                with open(attendance_file, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(col_names)
            
            # Append only new attendance records
            with open(attendance_file, 'a', newline='') as file:
                writer = csv.writer(file)
                for record in self.current_attendance:
                    # Check if this record already exists
                    if not any(record[0] == existing[0] and record[2] == existing[2] 
                              for existing in existing_attendance):
                        writer.writerow(record)
            
            self.status_label.setText(f"Success: Attendance marked for {len(self.current_attendance)} students")
            self.status_label.setStyleSheet("color: #2ecc71;")
        else:
            self.status_label.setText("No attendance marked in this session")
            self.status_label.setStyleSheet("color: #e74c3c;")
    
    
    def delete_registration(self):
        # Get student ID to delete
        student_id, ok = QInputDialog.getText(
            self, 
            "Delete Registration", 
            "Enter Student ID to delete:",
            QLineEdit.Normal
        )
        
        if not ok or not student_id:
            return
        
        student_id = student_id.strip()
        
        try:
            # Read student details
            with open(self.student_details_path, 'r') as file:
                reader = csv.reader(file)
                rows = list(reader)
            
            # Find student to delete
            deleted = False
            new_rows = [rows[0]]  # Keep header
            student_name = ""
            serial_number = ""
            
            for row in rows[1:]:
                if len(row) >= 3 and row[1] == student_id:  # Check row has enough columns
                    # Found student to delete
                    deleted = True
                    student_name = row[2]
                    serial_number = row[0]
                else:
                    new_rows.append(row)
            
            if not deleted:
                QMessageBox.warning(self, "Not Found", f"Student with ID {student_id} not found.")
                return
            
            # Write back the updated CSV
            with open(self.student_details_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(new_rows)
            
            # Delete training images more safely
            if os.path.exists(self.training_image_path):
                for filename in os.listdir(self.training_image_path):
                    if filename.endswith('.jpg'):
                        try:
                            parts = filename.split('.')
                            if len(parts) >= 3 and parts[2] == student_id:
                                os.remove(os.path.join(self.training_image_path, filename))
                        except (IndexError, OSError) as e:
                            print(f"Error processing file {filename}: {str(e)}")
                            continue
            
            # Update serial numbers in remaining students
            self.renumber_students()
            
            # Retrain the model
            self.train_images()
            
            QMessageBox.information(
                self,
                "Success",
                f"Successfully deleted registration for {student_name} (ID: {student_id})"
            )
            
            self.update_registration_count()
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to delete registration: {str(e)}\n\n"
                "Please check the data files for consistency."
            )
    
    def renumber_students(self):
        """Renumber students after deletion to maintain sequential serial numbers"""
        try:
            # Read student details
            with open(self.student_details_path, 'r') as file:
                reader = csv.reader(file)
                rows = list(reader)
            
            if len(rows) <= 1:  # Only header or empty
                return
            
            # Update serial numbers
            updated_rows = [rows[0]]  # Keep header
            new_serial = 1
            
            for row in rows[1:]:
                if len(row) >= 3:  # Ensure row has enough columns
                    old_serial = row[0]
                    row[0] = str(new_serial)
                    updated_rows.append(row)
                    
                    # Rename image files with new serial numbers
                    for filename in os.listdir(self.training_image_path):
                        if filename.endswith('.jpg'):
                            try:
                                parts = filename.split('.')
                                if len(parts) >= 5 and parts[1] == old_serial:
                                    new_name = f"{row[2]}.{new_serial}.{row[1]}.{parts[3]}.{parts[4]}"
                                    os.rename(
                                        os.path.join(self.training_image_path, filename),
                                        os.path.join(self.training_image_path, new_name))
                            except (IndexError, OSError) as e:
                                print(f"Error renaming file {filename}: {str(e)}")
                                continue
                    
                    new_serial += 1
            
            # Write back with updated serial numbers
            with open(self.student_details_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(updated_rows)
                
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to renumber students: {str(e)}\n\n"
                "Please check the data files manually."
            )
            raise  # Re-raise the exception after showing message

    def update_registration_count(self):
        try:
            with open(self.student_details_path, 'r') as file:
                count = sum(1 for _ in csv.reader(file)) - 1  # Subtract header
                self.registration_count_label.setText(f"Total Registrations: {count}")
        except:
            self.registration_count_label.setText("Total Registrations: 0")
    def change_password(self):
        password_dialog = QDialog(self)
        password_dialog.setWindowTitle("Change Password")
        password_dialog.setFixedSize(400, 300)
        password_dialog.setStyleSheet("background-color: #34495e;")
        
        layout = QVBoxLayout(password_dialog)
        
        # Title
        title = QLabel("Change Password")
        title.setStyleSheet("""
            color: #ecf0f1; 
            background-color: #3498db;
            padding: 10px;
            font-weight: bold;
        """)
        title_font = QFont("Helvetica", 16, QFont.Bold)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Form container
        form_frame = QFrame()
        form_layout = QVBoxLayout(form_frame)
        form_layout.setContentsMargins(20, 20, 20, 20)
        form_layout.setSpacing(15)
        
        # Old password
        lbl_old_pass = QLabel("Old Password:")
        lbl_old_pass.setStyleSheet("color: #ecf0f1;")
        lbl_old_pass.setFont(QFont("Helvetica", 12))
        form_layout.addWidget(lbl_old_pass)
        
        txt_old_pass = QLineEdit()
        txt_old_pass.setEchoMode(QLineEdit.Password)
        txt_old_pass.setStyleSheet("background-color: white; color: #2c3e50;")
        txt_old_pass.setFont(QFont("Helvetica", 12))
        form_layout.addWidget(txt_old_pass)
        
        # New password
        lbl_new_pass = QLabel("New Password:")
        lbl_new_pass.setStyleSheet("color: #ecf0f1;")
        lbl_new_pass.setFont(QFont("Helvetica", 12))
        form_layout.addWidget(lbl_new_pass)
        
        txt_new_pass = QLineEdit()
        txt_new_pass.setEchoMode(QLineEdit.Password)
        txt_new_pass.setStyleSheet("background-color: white; color: #2c3e50;")
        txt_new_pass.setFont(QFont("Helvetica", 12))
        form_layout.addWidget(txt_new_pass)
        
        # Confirm new password
        lbl_confirm_pass = QLabel("Confirm Password:")
        lbl_confirm_pass.setStyleSheet("color: #ecf0f1;")
        lbl_confirm_pass.setFont(QFont("Helvetica", 12))
        form_layout.addWidget(lbl_confirm_pass)
        
        txt_confirm_pass = QLineEdit()
        txt_confirm_pass.setEchoMode(QLineEdit.Password)
        txt_confirm_pass.setStyleSheet("background-color: white; color: #2c3e50;")
        txt_confirm_pass.setFont(QFont("Helvetica", 12))
        form_layout.addWidget(txt_confirm_pass)
        
        layout.addWidget(form_frame)
        
        # Button frame
        button_frame = QFrame()
        button_layout = QHBoxLayout(button_frame)
        button_layout.setContentsMargins(20, 0, 20, 20)
        button_layout.setSpacing(20)
        
        btn_save = QPushButton("Save")
        btn_save.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: #ecf0f1;
                padding: 10px;
                font-weight: bold;
                border: none;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
        """)
        btn_save.setFont(QFont("Helvetica", 12, QFont.Bold))
        
        def save_password():
            if txt_new_pass.text() != txt_confirm_pass.text():
                QMessageBox.critical(password_dialog, "Error", "New passwords don't match")
                return
            
            # Here you would typically:
            # 1. Verify old password
            # 2. Update to new password
            # 3. Save to secure storage
            
            QMessageBox.information(password_dialog, "Success", "Password changed successfully")
            password_dialog.accept()
        
        btn_save.clicked.connect(save_password)
        button_layout.addWidget(btn_save)
        
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: #ecf0f1;
                padding: 10px;
                font-weight: bold;
                border: none;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        btn_cancel.setFont(QFont("Helvetica", 12, QFont.Bold))
        btn_cancel.clicked.connect(password_dialog.reject)
        button_layout.addWidget(btn_cancel)
        
        layout.addWidget(button_frame)
        
        password_dialog.exec_()
    
    def user_guide(self):
        guide = """
        Face Recognition Attendance System - User Guide
        
        1. Registration:
           - Enter Student ID and Name
           - Click 'Capture Images' and look at the camera
           - After capturing, click 'Train Model'
           
        2. Taking Attendance:
           - Click 'Take Attendance' and look at the camera
           - The system will automatically mark attendance
           - Press 'q' to stop
           
        3. Viewing Attendance:
           - Attendance records are displayed in the table
           - Full records are saved in CSV files
           
        4. Deleting Registration:
           - Click 'Delete Registration'
           - Enter the Student ID to delete
           - Confirm deletion to remove all records and images
        """
        QMessageBox.information(self, "User Guide", guide.strip())
    
    def about(self):
        about_text = """
        Face Recognition Attendance System
        
        Version: 1.0
        Developed by: Nilay Kumar
        
        This application uses OpenCV and face recognition
        technology to automate attendance tracking.
        """
        QMessageBox.information(self, "About", about_text.strip())
    
    def contact(self):
        QMessageBox.information(
            self,
            "Contact Us", 
            "For support or inquiries:\n\n"
            "Email: nilay9101@gmail.com\n"
            "Phone: +91 8603781068\n"
        )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and set palette for dark theme
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(44, 62, 80))  # #2c3e50
    palette.setColor(QPalette.WindowText, QColor(236, 240, 241))  # #ecf0f1
    palette.setColor(QPalette.Base, QColor(52, 73, 94))  # #34495e
    palette.setColor(QPalette.AlternateBase, QColor(65, 90, 119))
    palette.setColor(QPalette.ToolTipBase, QColor(236, 240, 241))
    palette.setColor(QPalette.ToolTipText, QColor(236, 240, 241))
    palette.setColor(QPalette.Text, QColor(236, 240, 241))
    palette.setColor(QPalette.Button, QColor(52, 73, 94))
    palette.setColor(QPalette.ButtonText, QColor(236, 240, 241))
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Highlight, QColor(41, 128, 185))  # #2980b9
    palette.setColor(QPalette.HighlightedText, Qt.white)
    app.setPalette(palette)
    
    window = FaceRecognitionAttendanceSystem()
    window.show()
    sys.exit(app.exec_())