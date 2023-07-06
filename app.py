import streamlit as st 
from streamlit_option_menu import option_menu
from PIL import Image
import tensorflow as tf
import os

# EDA Pkgs
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler

import joblib 



gender_dict = {"male":0,"female":1}

def get_value(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return value 

def get_key(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return key

# Load ML Models
def load_model(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model

def load_keras_model(model_file):
    loaded_model = tf.keras.models.load_model(os.path.join(model_file))
    return loaded_model

X_train = joblib.load('model/X_train_scaled.pkl')
y_train = joblib.load('model/y_train.pkl')
scaler = joblib.load('model/sc.pkl')

X_train_bbtb = joblib.load('model/X_train_BBTB.pkl')
y_train_bbtb = joblib.load('model/y_train_BBTB.pkl')
scaler_bbtb = joblib.load('model/sc_BBTB.pkl')

X_train_tbu = joblib.load('model/X_train_TBU.pkl')
y_train_tbu = joblib.load('model/y_train_TBU.pkl')
scaler_tbu = joblib.load('model/sc_TBU.pkl')


html_temp = """
		<div style="background-color:blue;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Prediksi Status Gizi Balita Menggunakan Machine Learning </h1>
		</div>
		"""

descriptive_message_temp ="""
	<div style="background-color:beige;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<h3 style="text-align:justify;color:black;padding:10px">Definisi Status Gizi</h3>
		<p>Status gizi adalah kondisi tubuh akibat konsumsi makanan dan penggunaan zat gizi. Nutrisi dibutuhkan oleh tubuh sebagai sumber energi, untuk pertumbuhan dan pemeliharaan jaringan tubuh, serta untuk mengontrol fungsi tubuh.</p>
	</div>
	"""

descriptive_message_temp2 ="""
	<div style="background-color:beige;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<h3 style="text-align:justify;color:black;padding:10px">Indeks Standar Antropometri Anak</h3>
		<p>Standar Antropometri Anak digunakan untuk menilai atau menentukan status gizi anak. Penilaian status gizi Anak dilakukan dengan membandingkan hasil pengukuran berat badan dan panjang/tinggi badan dengan Standar Antropometri Anak. Klasifikasi penilaian status gizi berdasarkan Indeks Antropometri sesuai dengan kategori status gizi pada WHO Child Growth Standards untuk anak usia 0-5 tahun dan The WHO Reference 2007 untuk anak 5-18 tahun.</p>
		<p>Standar Antropometri Anak didasarkan pada parameter berat badan dan panjang/tinggi badan yang terdiri atas 4 (empat) indeks, meliputi <br/> - Indeks Berat Badan menurut umur BB/U <br/> - Indeks Panjang Badan menurut Umur atau Tinggi Badan menurut 
Umur (PB/U atau TB/U) <br/> - Indeks Berat Badan menurut Panjang Badan/Tinggi Badan (BB/PB 
atau BB/TB) <br/> - Indeks Masa Tubuh menurut Umur (IMT/U)</p>
		<p>Berikut adalah tabel kategori dan ambang batas status gizi anak<p>
	</div>
	"""

html = """
	<div style="background-color:beige;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<h3 style="text-align:justify;color:black;padding:10px">Machine Learning untuk Prediksi Status Gizi Balita</h3>
		<p>Pada website ini, proses klasifikasi akan dilakukan dengan menggunakan Algoritma K-Nearest Neighbors dan Artificial Neural Network.</p>
		<p>Algoritma K-Nearest Neighbors merupakan sebuah algoritma yang digunakan untuk melakukan klasifikasi terhadap suatu objek berdasarkan kedekatan lokasi (jarak) suatu data dengan k tetangga terdekatnya pada data latih. Syarat nilai k adalah tidak boleh lebih besar dari jumlah data latih, dan nilai k harus ganjil dan lebih dari satu.</p>
		<p>Algoritma Artificial Neural Network atau Jaringan Syaraf Tiruan (JST) adalah sistem komputasi dimana arsitektur dan operasi diilhami dari pengetahuan tentang sel syaraf biologi dalam otak.<p>
	</div>

"""


def load_image(img):
	im =Image.open(os.path.join(img))
	return im


def main():

	st.markdown(html_temp,unsafe_allow_html=True)

	menu = ["Home","Prediksi"]
	sub_menu = ["Berat Badan/Umur","Berat Badan/Tinggi Badan","Tinggi Badan/Umur"]

	choice = st.sidebar.selectbox("Menu",menu)
	if choice == "Home":
		st.markdown(descriptive_message_temp,unsafe_allow_html=True)
		st.markdown(descriptive_message_temp2,unsafe_allow_html=True)
		st.image(load_image('gizi.png'), use_column_width=True)
		st.markdown(html,unsafe_allow_html=True)
	
	elif choice == "Prediksi":
		st.subheader("Predictive Analytics")
		activity = st.selectbox("Activity", sub_menu)
		if activity == "Berat Badan/Umur":
				age = st.number_input("Age (month)",0,60)
				weight = st.number_input("Weight (kg)")
				sex = st.radio("Sex",tuple(gender_dict.keys()))
				feature_list = [age,weight,get_value(sex,gender_dict)]
				
				single_sample = np.array(feature_list).reshape(1,-1)
				single_sample_scaled = scaler.transform(single_sample)

				model_choice = st.selectbox("Select Model",["KNN k=3","KNN k=5","KNN k=7","ANN"])

				if st.button("Predict"):
						if model_choice == "KNN k=3":
							loaded_model = load_model("model/knnBBU_k3.pkl")
							prediction = loaded_model.predict(single_sample_scaled)


						elif model_choice == "KNN k=5":
							loaded_model = load_model("model/knnBBU_k5.pkl")
							prediction = loaded_model.predict(single_sample_scaled)


						elif model_choice == "KNN k=7":
							loaded_model = load_model("model/knnBBU_k7.pkl")
							prediction = loaded_model.predict(single_sample_scaled)
							

						else:
							loaded_model = load_keras_model("model/annBBU_ROS.h5")
							predicted_class = loaded_model.predict(single_sample_scaled)
							prediction = np.argmax(predicted_class)
						
						if prediction == 0:
							st.success("Sangat Kurang")

						elif prediction == 1:
							st.success("Kurang")
						
						elif prediction == 2:
							st.success("Berat Badan Normal")
						else:
							st.success("Risiko Lebih")
				
		elif activity == "Berat Badan/Tinggi Badan":
				weight = st.number_input("Weight (kg)")
				height = st.number_input("Height (cm)")
				sex = st.radio("Sex",tuple(gender_dict.keys()))
				feature_list = [weight,height,get_value(sex,gender_dict)]
				
				single_sample = np.array(feature_list).reshape(1,-1)
				single_sample_scaled = scaler_bbtb.transform(single_sample)

				model_choice = st.selectbox("Select Model",["KNN k=3","KNN k=5","KNN k=7","ANN"])

				if st.button("Predict"):
						if model_choice == "KNN k=3":
							loaded_model = load_model("model/knnBBTB_k=3.pkl")
							prediction = loaded_model.predict(single_sample_scaled)


						elif model_choice == "KNN k=5":
							loaded_model = load_model("model/knnBBTB_k=5.pkl")
							prediction = loaded_model.predict(single_sample_scaled)


						elif model_choice == "KNN k=7":
							loaded_model = load_model("model/knnBBTB_k=7.pkl")
							prediction = loaded_model.predict(single_sample_scaled)
							

						else:
							loaded_model = load_keras_model("model/annBBTB_ROS.h5")
							predicted_class = loaded_model.predict(single_sample_scaled)
							prediction = np.argmax(predicted_class)
						
						if prediction == 0:
							st.success("Gizi Buruk")

						elif prediction == 1:
							st.success("Gizi Kurang")
						
						elif prediction == 2:
							st.success("Gizi Baik")

						elif prediction == 3:
							st.success("Risiko Gizi Lebih")
							
						elif prediction == 4:
							st.success("Gizi Lebih")
						else:
							st.success("Obesitas")

		elif activity == "Tinggi Badan/Umur":
				height = st.number_input("Height (cm)")
				age = st.number_input("Age (Month)",0,60)
				sex = st.radio("Sex",tuple(gender_dict.keys()))
				feature_list = [height,age,get_value(sex,gender_dict)]
				
				single_sample = np.array(feature_list).reshape(1,-1)
				single_sample_scaled = scaler_tbu.transform(single_sample)

				model_choice = st.selectbox("Select Model",["KNN k=3","KNN k=5","KNN k=7","ANN"])

				if st.button("Predict"):
						if model_choice == "KNN k=3":
							loaded_model = load_model("model/knnTBU_k3.pkl")
							prediction = loaded_model.predict(single_sample_scaled)


						elif model_choice == "KNN k=5":
							loaded_model = load_model("model/knnTBU_k5.pkl")
							prediction = loaded_model.predict(single_sample_scaled)


						elif model_choice == "KNN k=7":
							loaded_model = load_model("model/knnTBU_k7.pkl")
							prediction = loaded_model.predict(single_sample_scaled)
							

						else:
							loaded_model = load_keras_model("model/annTBU_ROS.h5")
							predicted_class = loaded_model.predict(single_sample_scaled)
							prediction = np.argmax(predicted_class)
						
						if prediction == 0:
							st.success("Sangat Pendek")

						elif prediction == 1:
							st.success("Pendek")
						
						else:
							st.success("Normal")

if __name__ == '__main__':
	main()


