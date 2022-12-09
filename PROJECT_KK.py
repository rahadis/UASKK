import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score

st.write("""<h1>Aplikasi Clustering Penduduk Berdasarkan Usia Menggunakan K-Medoids</h1>""",unsafe_allow_html=True)

with st.container():
    with st.sidebar:
        selected = option_menu(
        st.write("""<h3 style = "text-align: center;"><img src="https://www.analyticsvidhya.com/wp-content/uploads/2016/11/clustering.png" width="120" height="100"><br> 
        Kelompok 4 <p>Kecerdasan Komputasional</p></h3>""",unsafe_allow_html=True), 
        ["Home", "Data", "Prepocessing", "Clustering"], 
            icons=['house', 'file-earmark-font', 'bar-chart', 'gear', 'arrow-down-square', 'check2-square'], menu_icon="cast", default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "#33ff99", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px"},
                "nav-link-selected": {"background-color": "Gold"},
            }
        )

    if selected == "Home":
        img = Image.open('peta.jpg')
        st.image(img, use_column_width=False, width=500)

        st.write("""
        Badan Pusat Statistik (BPS) Kabupaten Bangkalan merilis hasil Sensus Penduduk Tahun 2020 per September 2020. Sesuai data yang dipublikasikan pada 26 Januari 2020, bahwa populasi penduduk Kabupaten Bangkalan sebanyak 1.060.377 jiwa.

Agip Yunaidi Solichin, Koordinator Fungsi Statistik Sosial BPS Bangkalan, mengatakan jumlah tersebut  meningkat sebanyak 153.616 jiwa (16,95%) dari jumlah penduduk 10 tahun yang lalu (2010) yang tercatat sebanyak 906.761 jiwa.
        """)

    elif selected == "Data":
        st.subheader("""Dataset Penduduk Kamal""")
        df = pd.read_csv('https://raw.githubusercontent.com/Ihsan210702/dataset/main/data-kamal.csv')
        st.dataframe(df)
        st.subheader("""Dataset Penduduk Kwanyar""")
        df = pd.read_csv('https://raw.githubusercontent.com/Ihsan210702/dataset/main/data-kwanyar.csv')
        st.dataframe(df)
        st.subheader("""Dataset Penduduk Labang""")
        df = pd.read_csv('https://raw.githubusercontent.com/Ihsan210702/dataset/main/data-labang.csv')
        st.dataframe(df)
        st.subheader("""Dataset Penduduk Tanah Merah""")
        df = pd.read_csv('https://raw.githubusercontent.com/Ihsan210702/dataset/main/data-tanah-merah.csv')
        st.dataframe(df)
        st.subheader("""Dataset Penduduk Bangkalan""")
        df = pd.read_csv('https://raw.githubusercontent.com/Ihsan210702/dataset/main/data-bangkalan.csv')
        st.dataframe(df)
        st.subheader("""Dataset Penduduk Socah""")
        df = pd.read_csv('https://raw.githubusercontent.com/Ihsan210702/dataset/main/data-socah.csv')
        st.dataframe(df)
    elif selected == "Prepocessing":
        st.subheader("""Normalisasi Data""")
        st.write("""Rumus Normalisasi Data :""")
        st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
        #Read Dataset
        df = pd.read_csv('https://raw.githubusercontent.com/Ihsan210702/dataset/main/data-kamal.csv')
        df1 = pd.read_csv('https://raw.githubusercontent.com/Ihsan210702/dataset/main/data-kwanyar.csv')
        df2 = pd.read_csv('https://raw.githubusercontent.com/Ihsan210702/dataset/main/data-labang.csv')
        df3 = pd.read_csv('https://raw.githubusercontent.com/Ihsan210702/dataset/main/data-tanah-merah.csv')
        df4 = pd.read_csv('https://raw.githubusercontent.com/Ihsan210702/dataset/main/data-bangkalan.csv')
        df5 = pd.read_csv('https://raw.githubusercontent.com/Ihsan210702/dataset/main/data-socah.csv')
        st.markdown("""
        Dimana :
        - X = data yang akan dinormalisasi atau data asli
        - min = nilai minimum semua data asli
        - max = nilai maksimum semua data asli
        """)
        #Mendefinisikan Varible X dan Y
        X = df.drop(columns=['Desa'])
        X1 = df1.drop(columns=['Desa'])
        X2 = df2.drop(columns=['Desa'])
        X3 = df3.drop(columns=['Desa'])
        X4 = df4.drop(columns=['Desa'])
        X5 = df5.drop(columns=['Desa'])

        #NORMALISASI NILAI KAMAL
        scaler = MinMaxScaler()
        #scaler.fit(features)
        #scaler.transform(features)
        scaled = scaler.fit_transform(X)
        features_names = X.columns.copy()
        #features_names.remove('label')
        kamal = pd.DataFrame(scaled, columns=features_names)

        #NORMALISASI NILAI KWANYAR
        scaler = MinMaxScaler()
        #scaler.fit(features)
        #scaler.transform(features)
        scaled = scaler.fit_transform(X1)
        features_names = X.columns.copy()
        #features_names.remove('label')
        kwanyar = pd.DataFrame(scaled, columns=features_names)

        #NORMALISASI NILAI LABANG
        scaler = MinMaxScaler()
        #scaler.fit(features)
        #scaler.transform(features)
        scaled = scaler.fit_transform(X2)
        features_names = X.columns.copy()
        #features_names.remove('label')
        labang = pd.DataFrame(scaled, columns=features_names)

        #NORMALISASI NILAI TANAH MERAH
        scaler = MinMaxScaler()
        #scaler.fit(features)
        #scaler.transform(features)
        scaled = scaler.fit_transform(X3)
        features_names = X.columns.copy()
        #features_names.remove('label')
        merah = pd.DataFrame(scaled, columns=features_names)

        #NORMALISASI NILAI BANGKALAN
        scaler = MinMaxScaler()
        #scaler.fit(features)
        #scaler.transform(features)
        scaled = scaler.fit_transform(X4)
        features_names = X.columns.copy()
        #features_names.remove('label')
        bangkalan = pd.DataFrame(scaled, columns=features_names)

        #NORMALISASI NILAI SOCAH
        scaler = MinMaxScaler()
        #scaler.fit(features)
        #scaler.transform(features)
        scaled = scaler.fit_transform(X5)
        features_names = X.columns.copy()
        #features_names.remove('label')
        socah = pd.DataFrame(scaled, columns=features_names)

        st.subheader('Hasil Normalisasi Data Kamal')
        st.write(kamal)
        st.subheader('Hasil Normalisasi Data Kwanyar')
        st.write(kwanyar)
        st.subheader('Hasil Normalisasi Data Labang')
        st.write(labang)
        st.subheader('Hasil Normalisasi Data Tanah Merah')
        st.write(merah)
        st.subheader('Hasil Normalisasi Data Bangkalan')
        st.write(bangkalan)
        st.subheader('Hasil Normalisasi Data Socah')
        st.write(socah)
    elif selected =="Clustering":
        with st.form("my_form"):
            st.subheader("Implementasi Clustering Penduduk Wilayah Kabupaten Bangkalan")
            Clustering = st.slider('Jumlah Cluster', 2, 5)
            model = st.selectbox('Pilihlah Kecamatan yang akan di cluster',
                        ('Kamal', 'Kwanyar', 'Labang', 'Tanah Merah', 'Bangkalan', 'Socah'))
            prediksi = st.form_submit_button("Submit")
            if prediksi:
                inputs = np.array([
                    Clustering
                ])
                df = pd.read_csv('https://raw.githubusercontent.com/Ihsan210702/dataset/main/data-kamal.csv')
                X = df.drop(columns=['Desa'])
                y = df['Desa'].values
                df1 = pd.read_csv('https://raw.githubusercontent.com/Ihsan210702/dataset/main/data-kwanyar.csv')
                X1 = df1.drop(columns=['Desa'])
                y1 = df1['Desa'].values
                df2 = pd.read_csv('https://raw.githubusercontent.com/Ihsan210702/dataset/main/data-labang.csv')
                X2= df2.drop(columns=['Desa'])
                y2 = df2['Desa'].values
                df3 = pd.read_csv('https://raw.githubusercontent.com/Ihsan210702/dataset/main/data-tanah-merah.csv')
                X3 = df3.drop(columns=['Desa'])
                y3 = df3['Desa'].values
                df4 = pd.read_csv('https://raw.githubusercontent.com/Ihsan210702/dataset/main/data-bangkalan.csv')
                X4 = df4.drop(columns=['Desa'])
                y4 = df4['Desa'].values
                df5 = pd.read_csv('https://raw.githubusercontent.com/Ihsan210702/dataset/main/data-socah.csv')
                X5 = df5.drop(columns=['Desa'])
                y5 = df5['Desa'].values
                
                df_min = X.min()
                df_max = X.max()

                #NORMALISASI DATA KAMAL
                scaler = MinMaxScaler()
                #scaler.fit(features)
                #scaler.transform(features)
                scaled = scaler.fit_transform(X)
                features_names = X.columns.copy()
                #features_names.remove('label')
                kamal = pd.DataFrame(scaled, columns=features_names)

                #NORMALISASI DATA KWANYAR
                scaler = MinMaxScaler()
                #scaler.fit(features)
                #scaler.transform(features)
                scaled = scaler.fit_transform(X1)
                features_names = X.columns.copy()
                #features_names.remove('label')
                kwanyar = pd.DataFrame(scaled, columns=features_names)

                #NORMALISASI DATA LABANG
                scaler = MinMaxScaler()
                #scaler.fit(features)
                #scaler.transform(features)
                scaled = scaler.fit_transform(X2)
                features_names = X.columns.copy()
                #features_names.remove('label')
                labang = pd.DataFrame(scaled, columns=features_names)

                #NORMALISASI DATA TANAH MERAH
                scaler = MinMaxScaler()
                #scaler.fit(features)
                #scaler.transform(features)
                scaled = scaler.fit_transform(X3)
                features_names = X.columns.copy()
                #features_names.remove('label')
                merah = pd.DataFrame(scaled, columns=features_names)

                #NORMALISASI DATA BANGKALAN
                scaler = MinMaxScaler()
                #scaler.fit(features)
                #scaler.transform(features)
                scaled = scaler.fit_transform(X4)
                features_names = X.columns.copy()
                #features_names.remove('label')
                bangkalan = pd.DataFrame(scaled, columns=features_names)

                #NORMALISASI DATA SOCAH
                scaler = MinMaxScaler()
                #scaler.fit(features)
                #scaler.transform(features)
                scaled = scaler.fit_transform(X5)
                features_names = X.columns.copy()
                #features_names.remove('label')
                socah = pd.DataFrame(scaled, columns=features_names)
                if model == 'Kamal':
                    desa = pd.DataFrame({'Desa': y})
                    cluster_K = KMedoids (n_clusters=Clustering,metric="euclidean",init="random")
                    fitted_K = cluster_K.fit_predict(kamal)

                    cluster = pd.DataFrame({'Cluster': fitted_K})
                    result = pd.concat([desa, cluster], axis=1)
                    result
                    st.success(f'Silhouette Score: {silhouette_score(kamal, fitted_K)}')
                if model == 'Kwanyar':
                    desa = pd.DataFrame({'Desa': y1})
                    cluster_K = KMedoids (n_clusters=Clustering,metric="euclidean",init="random")
                    fitted_K = cluster_K.fit_predict(kwanyar)

                    cluster = pd.DataFrame({'Cluster': fitted_K})
                    result = pd.concat([desa, cluster], axis=1)
                    result
                    st.success(f'Silhouette Score: {silhouette_score(kwanyar, fitted_K)}')
                if model == 'Labang':
                    desa = pd.DataFrame({'Desa': y2})
                    cluster_K = KMedoids (n_clusters=Clustering,metric="euclidean",init="random")
                    fitted_K = cluster_K.fit_predict(labang)

                    cluster = pd.DataFrame({'Cluster': fitted_K})
                    result = pd.concat([desa, cluster], axis=1)
                    result
                    st.success(f'Silhouette Score: {silhouette_score(labang, fitted_K)}')
                if model == 'Tanah Merah':
                    desa = pd.DataFrame({'Desa': y3})
                    cluster_K = KMedoids (n_clusters=Clustering,metric="euclidean",init="random")
                    fitted_K = cluster_K.fit_predict(merah)

                    cluster = pd.DataFrame({'Cluster': fitted_K})
                    result = pd.concat([desa, cluster], axis=1)
                    result
                    st.success(f'Silhouette Score: {silhouette_score(merah, fitted_K)}')
                if model == 'Bangkalan':
                    desa = pd.DataFrame({'Desa': y4})
                    cluster_K = KMedoids (n_clusters=Clustering,metric="euclidean",init="random")
                    fitted_K = cluster_K.fit_predict(bangkalan)

                    cluster = pd.DataFrame({'Cluster': fitted_K})
                    result = pd.concat([desa, cluster], axis=1)
                    result
                    st.success(f'Silhouette Score: {silhouette_score(bangkalan, fitted_K)}')
                if model == 'Socah':
                    desa = pd.DataFrame({'Desa': y5})
                    cluster_K = KMedoids (n_clusters=Clustering,metric="euclidean",init="random")
                    fitted_K = cluster_K.fit_predict(socah)

                    cluster = pd.DataFrame({'Cluster': fitted_K})
                    result = pd.concat([desa, cluster], axis=1)
                    result
                    st.success(f'Silhouette Score: {silhouette_score(socah, fitted_K)}')