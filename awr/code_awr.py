from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import inspect, os
import subprocess
import sys
from selenium.webdriver.support.ui import Select
import os
import datetime
import shutil
import time
import pandas as pd
from sklearn.cluster import DBSCAN
import math
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.colors
from IPython import display
import numpy as np
import csv
import datetime


chrome_options = Options()
chrome_options.add_argument(
    'load-extension={0}'.format(r"C:\Users\User\AppData\Local\Google\Chrome\User Data\Default\Extensions\pfillhniocmjcapelhjcianojmoidjdk\3.0.5_0")
)
download_directory = {'download.default_directory' : 'D:\\Azka\\ET\\Semester 7\\TA1 dan Seminar\\myta\\data\\test' }
chrome_options.add_experimental_option('prefs', download_directory)


driver = webdriver.Chrome(options=chrome_options)
driver.get('https://dev.ti.com/gallery/view/mmwavebeta/mmWave_Demo_Visualizer_Record/ver/3.1.0/')

def read_remove(Directory):
    time_mulai = datetime.datetime.now()
    dict_data = {}
    dict_data_baru = {}
    data = []
    key_data = 0
    counting = 0

    for dirpath, dirnames,filenames in os.walk(Directory):
        for file in filenames :         
            time_jalan = datetime.datetime.now()
            #read file
            dirfile = os.path.join(dirpath, file)
            df = pd.read_csv(dirfile, index_col=False)
            data = df.iloc[:, 3:5]
            #ambil kolom kecepatan
            data_kecepatan = df.iloc[:, 6]
            #array x y v (belum difilter)
            data_full = pd.concat([data, data_kecepatan], axis = 1).values

            if len(data) == 0:
                plt.scatter(0, 0, c='black', s=1)
            else:
                model = DBSCAN(eps=1, min_samples=30).fit(data)    
                colors = model.labels_
                list_DBSCAN = Counter(model.labels_)

                #dict index per cluster tanpa -1
                dict_to_counting = {}
                #index buat isi dict_to_counting
                index = 0

                #buat dict_to_counting : dict[integer cluster] = [array index]
                for integer in model.labels_:
                    if integer == -1:
                        pass
                    else :
                        if (integer in dict_to_counting.keys()):
                            dict_to_counting[integer].append(index)
                        else : 
                            dict_to_counting[integer] = [index]
                    index +=1    
				
                #print(dict_to_counting)
                counting = len(dict_to_counting)

                #deklarasi array array_to_plot
                array_to_plot = []

                #pembuatan array_to_plot : [Index, rataX, rataY, rataV]
                for index in dict_to_counting:
                    X = 0
                    Y = 0
                    V = 0
                    #assign variabel X, Y, V sebelum dirata-ratakan
                    for value in dict_to_counting[index]:
                        X += data_full[value][0]
                        Y += data_full[value][1]
                        V += data_full[value][2]

                    #rata-rata jarak X, jarak Y, kecepatan V
                    rataX= X /len(dict_to_counting[index])
                    rataY= Y /len(dict_to_counting[index])
                    rataV= str(round(V /len(dict_to_counting[index]),2)) + " m/s"

                    #assign array “array_to_plot”
                    array = [index, rataX, rataY, rataV]
                    array_to_plot.append(array)

				
                #print(array_to_plot)
                


                array_to_plot = pd.DataFrame(data= array_to_plot).values
                for i in range(len(array_to_plot)):
                    plt.scatter(array_to_plot[i][1], array_to_plot[i][2],  label=array_to_plot[i][3], s= 50 )

                

            #display.display(plt.gcf())
            fig = plt.gcf()
            plt.title(time_jalan)
            plt.legend()
            plt.ylim(0, 10)
            plt.xlim(-10, 10)
            fig.savefig(r'C:\Users\User\deep-learning-v2-pytorch\web\Flaskta\website\static\plot_awr.png')
            plt.clf()
            #display.clear_output(wait=True)
			
            

            with open (r'C:\Users\User\deep-learning-v2-pytorch\web\Flaskta\website\static\data_awr.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Time", "Target"])
                writer.writerow([time_jalan.strftime("%X"), counting])
            
            
            try : 
                dataframe = pd.read_csv(r'.\data_history_awr2.csv', index_col=False)
            except FileNotFoundError:
                with open(r'.\\data_history_awr2.csv', 'a', newline='') as csvhistory:
                    writer = csv.writer(csvhistory)
                    writer.writerow(["Time", "Target"])
                
        
            with open(r'.\\data_history_awr2.csv', 'a', newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow([time_jalan.strftime("%X"), counting])
                    
            dataframe = pd.read_csv(r'.\data_history_awr2.csv', index_col=False)
            now =  datetime.datetime.now().strftime("%H:%M:%S")
            if datetime.datetime.strptime(now, '%H:%M:%S') - datetime.datetime.strptime(dataframe.iloc[0][0], '%H:%M:%S') > datetime.timedelta(hours=24):
                 with open(r'.\\data_history_awr2.csv', 'w', newline='') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(["Time", "Target"])
                    writer.writerow([time_jalan.strftime("%X"), counting])
            

            file_modified = datetime.datetime.fromtimestamp(os.path.getmtime(dirfile))
            if datetime.datetime.now() - file_modified > datetime.timedelta(seconds=0.01):
                os.remove(dirfile)
                        
    return data


element = WebDriverWait(driver,30).until(EC.presence_of_element_located((By.ID, "cancelBtn")))


driver.find_element_by_id('cancelBtn').click()
driver.find_element_by_id('tab_1').click()


config = driver.find_element_by_xpath(".//*[@id='ti_widget_button_load_cfg']").click()
subprocess.Popen('tes.exe')

time.sleep(10)
config2 = driver.find_element_by_xpath(".//*[@id='ti_widget_button_load_cfg']").click()
subprocess.Popen('tes.exe')

time.sleep(10)


i = True
while i == True :
        driver.find_element_by_id('ti_widget_button_LOG').click()
        time.sleep(1)
        driver.find_element_by_id('ti_widget_button_LOG').click()
        time.sleep(1)
        read_remove("D:\\Azka\\ET\\Semester 7\\TA1 dan Seminar\\myta\\data\\test")



