import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import cv2
import numpy as np
import mysql.connector
import threading
from PIL import Image, ImageTk
import face_recognition
import queue
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

class FacialRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Recognition System")
        self.root.geometry("1000x600")
        self.root.configure(bg='#002460')
        self.cap = cv2.VideoCapture(0)
        self.frame = None
        self.processed_frame = None
        self.recognizing = False
        self.recognition_thread = None
        self.recognition_queue = queue.Queue()
        self.display_queue = queue.Queue()
        self.required_images = 20
        self.saving_images = False
        self.img_count = 0

        self.setup_layout()
        self.db_setup()

        self.update_video()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_layout(self):
        style = ttk.Style()
        style.configure("Custom.TLabelframe", background="#002460")
        style.configure("Custom.TLabelframe.Label", background="#002460", foreground="white")
        style.configure("Custom.TLabel", background="#002460", foreground="white")
        style.configure("Custom.TButton", background="#4caf50", foreground="black")
        style.map('Custom.TButton', background=[('active', '#000000')],
                  relief=[('pressed', 'flat'), ('!pressed', 'flat')])
        style.configure("Custom.TEntry", fieldbackground="#002460", foreground="black")  # Changed to black

        main_frame = ttk.Frame(self.root, padding="10", style="Custom.TLabelframe")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Video Feed Frame
        video_frame = ttk.LabelFrame(main_frame, text="Video Feed", style="Custom.TLabelframe")
        video_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        video_frame.grid_propagate(False)  # Disable automatic resizing of the frame

        self.video_label = ttk.Label(video_frame, text="Placeholder for video feed")
        self.video_label.pack(padx=20, pady=20)

        # Log Frame
        log_frame = ttk.LabelFrame(main_frame, text="Log", style="Custom.TLabelframe")
        log_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        log_frame.grid_propagate(False)  # Disable automatic resizing of the frame

        self.log_text = tk.Text(log_frame, wrap=tk.WORD, state=tk.DISABLED, bg="#002460", fg="white", width=60, height=10)
        self.log_text.pack(expand=True, fill=tk.BOTH)

        # Controls Frame
        controls_frame = ttk.LabelFrame(main_frame, text="Controls", style="Custom.TLabelframe")
        controls_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")

        # Widgets in Controls Frame
        ttk.Label(controls_frame, text="Name:", style="Custom.TLabel").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.name_entry = ttk.Entry(controls_frame, style="Custom.TEntry")
        self.name_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(controls_frame, text="CNIC:", style="Custom.TLabel").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.cnic_entry = ttk.Entry(controls_frame, style="Custom.TEntry")
        self.cnic_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(controls_frame, text="Email:", style="Custom.TLabel").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.email_entry = ttk.Entry(controls_frame, style="Custom.TEntry")
        self.email_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(controls_frame, text="Mobile No.:", style="Custom.TLabel").grid(row=3, column=0, padx=5, pady=5, sticky="e")
        self.mobile_entry = ttk.Entry(controls_frame, style="Custom.TEntry")
        self.mobile_entry.grid(row=3, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(controls_frame, text="Designation:", style="Custom.TLabel").grid(row=4, column=0, padx=5, pady=5, sticky="e")
        self.designation_entry = ttk.Entry(controls_frame, style="Custom.TEntry")
        self.designation_entry.grid(row=4, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(controls_frame, text="Age:", style="Custom.TLabel").grid(row=5, column=0, padx=5, pady=5, sticky="e")
        self.age_entry = ttk.Entry(controls_frame, style="Custom.TEntry")
        self.age_entry.grid(row=5, column=1, padx=5, pady=5, sticky="w")

        ttk.Button(controls_frame, text="Save Face", command=self.save_face, style="Custom.TButton").grid(row=6, column=0, columnspan=2, padx=5, pady=5)
        self.photo_count_label = ttk.Label(controls_frame, text="Photos Clicked: 0", style="Custom.TLabel")  # Define the photo count label
        self.photo_count_label.grid(row=7, column=0, columnspan=2, padx=5, pady=5)
        ttk.Button(controls_frame, text="Train Model", command=self.train_model, style="Custom.TButton").grid(row=8, column=0, columnspan=2, padx=5, pady=5)
        self.recognize_btn = ttk.Button(controls_frame, text="Recognize Faces", command=self.toggle_recognition, style="Custom.TButton")  # Define recognize button
        self.recognize_btn.grid(row=9, column=0, columnspan=2, padx=5, pady=5)

        # Configure weights for resizing
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)

    def db_setup(self):
        self.conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="facial_recognition"
        )
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                encoding BLOB NOT NULL,
                email VARCHAR(255) NOT NULL,
                cnic VARCHAR(20) NOT NULL,
                mobile VARCHAR(20) NOT NULL,
                designation VARCHAR(100) NOT NULL,
                age INT NOT NULL
            )
        ''')
        self.conn.commit()

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame.copy()

            # Update video display with either the processed frame or the captured frame
            cv2image = cv2.cvtColor(self.processed_frame if self.processed_frame is not None else self.frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        # Keep the update_video function running
        self.root.after(10, self.update_video)

    def align_face(self, frame, face_location):
        top, right, bottom, left = face_location
        face_image = frame[top:bottom, left:right]
        landmarks = face_recognition.face_landmarks(face_image)

        if landmarks:
            # Use landmarks to align face
            left_eye = landmarks[0]['left_eye']
            right_eye = landmarks[0]['right_eye']
            left_eye_center = np.mean(left_eye, axis=0).astype("int")
            right_eye_center = np.mean(right_eye, axis=0).astype("int")

            # Calculate angle between the eye centroids
            dy = right_eye_center[1] - left_eye_center[1]
            dx = right_eye_center[0] - left_eye_center[0]
            angle = np.degrees(np.arctan2(dy, dx)) - 180

            # Get rotation matrix
            eyes_center = ((left_eye_center[0] + right_eye_center[0]) / 2,
                           (left_eye_center[1] + right_eye_center[1]) / 2)
            M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1)

            # Rotate the image
            (h, w) = face_image.shape[:2]
            aligned_face = cv2.warpAffine(face_image, M, (w, h), flags=cv2.INTER_CUBIC)
            return aligned_face

        return face_image  # Return original face if alignment fails

    def normalize_embedding(self, embedding):
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

    def get_face_encoding(self, frame):
        face_locations = face_recognition.face_locations(frame)
        if face_locations:
            aligned_faces = [self.align_face(frame, loc) for loc in face_locations]
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            return self.normalize_embedding(face_encodings[0]) if face_encodings else None
        return None

    def save_face(self):
        if self.saving_images:
            return  # Prevent multiple concurrent save operations

        self.saving_images = True
        self.img_count = 0
        self.photo_count_label.config(text=f"Photos Clicked: {self.img_count}")

        name = self.name_entry.get()
        email = self.email_entry.get()
        cnic = self.cnic_entry.get()
        mobile = self.mobile_entry.get()
        designation = self.designation_entry.get()
        age = self.age_entry.get()

        if not name or not email or not cnic or not mobile or not designation or not age:
            messagebox.showerror("Error", "All fields must be filled.")
            self.saving_images = False
            return

        # Validate email, CNIC, and mobile number
        if "@" not in email:
            messagebox.showerror("Error", "Invalid email address.")
            self.saving_images = False
            return

        if len(cnic) != 13 or not cnic.isdigit():
            messagebox.showerror("Error", "CNIC must be 13 digits.")
            self.saving_images = False
            return

        if len(mobile) != 11 or not mobile.isdigit():
            messagebox.showerror("Error", "Mobile number must be 11 digits.")
            self.saving_images = False
            return

        person_dir = os.path.join('data', name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)

        def capture_images():
            while self.img_count < self.required_images:
                if self.frame is not None:
                    img_path = os.path.join(person_dir, f"{name}_{self.img_count + 1}.jpg")
                    cv2.imwrite(img_path, self.frame)

                    face_encoding = self.get_face_encoding(self.frame)
                    if face_encoding is not None:
                        self.save_to_db(name, face_encoding, email, cnic, mobile, designation, age)

                    self.img_count += 1
                    self.photo_count_label.config(text=f"Photos Clicked: {self.img_count}")
                    time.sleep(1)  # Delay to capture images at intervals

            self.saving_images = False
            messagebox.showinfo("Info", f"{self.required_images} images and encodings saved successfully")

        threading.Thread(target=capture_images).start()


    def save_to_db(self, name, encoding, email, cnic, mobile, designation, age):
        encoding_blob = np.array(encoding).tobytes()  # Convert encoding to bytes
        self.cursor.execute('''
            INSERT INTO users (name, encoding, email, cnic, mobile, designation, age) 
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        ''', (name, encoding_blob, email, cnic, mobile, designation, age))
        self.conn.commit()

    def train_model(self):
        data_dir = 'data'
        images, labels, label_names = self.load_data(data_dir)

        if len(label_names) == 0:
            messagebox.showerror("Error", "No data to train on.")
            return

        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)
        labels_encoded = to_categorical(labels_encoded, num_classes=len(label_names))

        X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

        model = self.create_model(len(label_names))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # More aggressive data augmentation
        datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            horizontal_flip=True,
            brightness_range=[0.7, 1.3],
            shear_range=0.3,
            zoom_range=0.3,
            fill_mode='nearest'
        )

        model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=30, validation_data=(X_test, y_test))

        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
        messagebox.showinfo("Model Accuracy", f"Model accuracy: {test_acc * 100:.2f}%")

        model.save('face_recognition_model.keras')  # Save in the recommended format
        with open('label_names.npy', 'wb') as f:
            np.save(f, le.classes_)

    def create_model(self, num_classes):
        model = Sequential([
            Input(shape=(64, 64, 1)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.3),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.3),
            Conv2D(256, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.4),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        return model

    def load_data(self, data_dir):
        images, labels, label_names = [], [], []

        for label in os.listdir(data_dir):
            label_path = os.path.join(data_dir, label)
            if os.path.isdir(label_path):
                for img_name in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (64, 64))  # Ensure all images are 64x64
                        images.append(img)
                        labels.append(label)
                label_names.append(label)

        images = np.array(images)
        images = images.reshape(images.shape[0], 64, 64, 1)  # Ensure the shape matches input shape of model
        images = images.astype('float32') / 255.0

        return images, labels, label_names

    def toggle_recognition(self):
        self.recognizing = not self.recognizing
        if self.recognizing:
            self.recognize_btn.config(text="Stop Recognizing")
            if self.recognition_thread is None or not self.recognition_thread.is_alive():
                self.recognition_thread = threading.Thread(target=self.recognition_loop)
                self.recognition_thread.start()
        else:
            self.recognize_btn.config(text="Recognize Faces")
            self.processed_frame = None  # Clear processed frame to show normal video

    def recognition_loop(self):
        while self.recognizing:
            if self.frame is not None:
                self.perform_face_recognition(self.frame)
            time.sleep(0.1)  # Small delay to prevent excessive CPU usage

    def perform_face_recognition(self, frame):
        try:
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                attempts = 0
                while attempts < 3:
                    try:
                        self.cursor.execute('SELECT name, encoding, email, cnic, mobile, designation, age FROM users')
                        users = self.cursor.fetchall()  # Fetch all results to prevent command sync issues
                        break  # If fetchall() is successful, break out of the loop
                    except mysql.connector.Error as e:
                        if e.errno == 2013 or e.errno == 2006:  # Lost connection or MySQL server has gone away
                            attempts += 1
                            self.reconnect_to_db()
                        else:
                            raise e

                # Initialize default values for variables
                name, email, cnic, mobile, designation, age = "Unknown", "", "", "", "", 0
                matches = []
                for user in users:
                    name, encoding_blob, email, cnic, mobile, designation, age = user
                    known_encoding = np.frombuffer(encoding_blob, dtype=np.float64)  # Convert BLOB to NumPy array
                    distance = face_recognition.face_distance([known_encoding], face_encoding)
                    matches.append((name, email, cnic, mobile, designation, age, distance[0]))

                # Use the face with the smallest distance if within a tolerance threshold
                if matches:
                    best_match = min(matches, key=lambda x: x[6])
                    if best_match[6] < 0.6:  # Tolerance level for face recognition
                        name, email, cnic, mobile, designation, age = best_match[:6]

                color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
                label = "Intruder" if name == "Unknown" else name

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                if name == "Unknown":
                    self.log_detection(name, "N/A", "N/A", "N/A", "N/A", "N/A", intruder=True)
                else:
                    self.log_detection(name, cnic, age, designation, email, mobile, intruder=False)

            self.processed_frame = frame  # Update the processed frame
        except Exception as e:
            print(f"Error in perform_face_recognition: {e}")

    def log_detection(self, name, cnic, age, designation, email, mobile, intruder=False):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"Detected {'Intruder' if intruder else name} (CNIC: {cnic}, Age: {age}, Designation: {designation}, Email: {email}, Mobile: {mobile}) at {timestamp}\n"

        self.log_text.config(state=tk.NORMAL)
        tag_name = 'intruder' if intruder else 'recognized'
        self.log_text.insert(tk.END, log_message, tag_name)
        self.log_text.tag_config('intruder', foreground='red')
        self.log_text.tag_config('recognized', foreground='green')
        self.log_text.config(state=tk.DISABLED)
        self.log_text.yview(tk.END)

    def reconnect_to_db(self):
        try:
            self.conn.ping(reconnect=True, attempts=3, delay=5)
        except mysql.connector.Error as e:
            print(f"Error reconnecting to database: {e}")
            # Optionally, handle the error further, e.g., alert the user, retry, etc.

    def on_closing(self):
        self.cap.release()
        self.conn.close()
        self.root.destroy()

    def __del__(self):
        self.cap.release()
        self.conn.close()

def show_splash_screen(root):
    splash = tk.Toplevel(root)
    splash.geometry("800x600")
    splash.title("Facial Recognition System")
    splash.configure(bg='#002460')

    style = ttk.Style()
    style.configure('TFrame', background='#002460')
    style.configure('TLabel', background='#002460', foreground='white', font=("Helvetica", 14))
    style.configure('TButton', background='black', foreground='black', font=("Helvetica", 12))
    style.map('TButton', background=[('active', '#000000')], relief=[('pressed', 'flat'), ('!pressed', 'flat')])

    splash_frame = ttk.Frame(splash, padding="10", style='TFrame')
    splash_frame.pack(fill=tk.BOTH, expand=True)

    title_label = ttk.Label(splash_frame, text="Artificial Intelligence Project", font=("Helvetica", 24, "bold"),
                            style='TLabel')
    title_label.pack(pady=30)

   

    title_label = ttk.Label(splash_frame, text="Facial Recognition System", font=("Helvetica", 24, "bold"),
                            style='TLabel')
    title_label.pack(pady=30)

    developed_by_label = ttk.Label(splash_frame, text="Developed by:", style='TLabel')
    developed_by_label.pack(pady=10)

    names_label = ttk.Label(splash_frame,
                            text="Zain Ahmed",
                            justify=tk.CENTER, style='TLabel')
    names_label.pack(pady=10)

    
    start_btn = ttk.Button(splash_frame, text="Start", command=lambda: start_app(root, splash), style='TButton')
    start_btn.pack(pady=30)

    def on_enter(e):
        start_btn.state(['pressed'])

    def on_leave(e):
        start_btn.state(['!pressed'])

    start_btn.bind("<Enter>", on_enter)
    start_btn.bind("<Leave>", on_leave)

def start_app(root, splash):
    splash.destroy()
    root.deiconify()  # Show the main window
    FacialRecognitionApp(root)

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide the main window while splash screen is open
    show_splash_screen(root)
    root.mainloop()
