# power consumption and solar generation are from temperature and humidity
import csv
import random
import numpy as np

# all months have 30 days

w_header = ["Outdoor Drybulb Temperature [C]", "Relative Humidity [%]", 
"Diffuse Solar Radiation [W/m2]", "Direct Solar Radiation [W/m2]", 
"6h Prediction Outdoor Drybulb Temperature [C]", "12h Prediction Outdoor Drybulb Temperature [C]", "24h Prediction Outdoor Drybulb Temperature [C]", 
"6h Prediction Relative Humidity [%]", "12h Prediction Relative Humidity [%]", "24h Prediction Relative Humidity [%]", 
"6h Prediction Diffuse Solar Radiation [W/m2]", "12h Prediction Diffuse Solar Radiation [W/m2]", "24h Prediction Diffuse Solar Radiation [W/m2]", 
"6h Prediction Direct Solar Radiation [W/m2]", "12h Prediction Direct Solar Radiation [W/m2]", "24h Prediction Direct Solar Radiation [W/m2]"
]

b_header = ["Month", "Hour", 
"Day Type", 
"Daylight Savings Status", "Indoor Temperature [C]", "Average Unmet Cooling Setpoint Difference [C]", "Indoor Relative Humidity [%]", 
"Equipment Electric Power [kWh]", 
"DHW Heating [kWh]", "Cooling Load [kWh]", "Heating Load [kWh]", 
"Solar Generation [W/kW]",
]

""" Parameters of weather """

temp_mean_mon = [[0, 1.9, 6.2, 11.5, 15.9, 19.8, 21.8, 21.5, 16.5, 10.9, 5.6, 1],
                 [1, 4, 10, 16, 21, 25, 27, 25, 21, 15, 8, 2],
                 [17, 17, 20, 23, 27, 29, 29, 29, 28, 26, 22, 18],
                 [0, 0, 0, 3, 6, 9, 11, 7, 4, 3, 2, 1],
                 [20, 20.5, 19, 16.4, 14.4, 12.1, 11.3, 11.9, 13.3, 14.6, 19.6, 15.65]]

rela_hum_mean = [[77, 70, 58, 49, 49, 50, 49, 51, 52, 62, 76, 80],
                 [66, 62, 67, 68, 68, 62, 71, 75, 79, 77, 74, 69],
                 [56.2, 69.9, 66.5, 68.2, 69.7, 70, 67.8, 67.5, 63.5, 57, 57.2, 52.8],
                 [74.4, 72, 72.6, 69.1, 70, 72, 76.3, 76.8, 75, 75.8, 75.7, 74.6],
                 [78, 80, 82, 83, 86, 86, 86, 83, 82, 77, 76, 77]]

def get_temp_day(month, month_day, b):
    if month_day < 15:
        last_month = month - 1
        return temp_mean_mon[b][month] + (temp_mean_mon[b][last_month] - temp_mean_mon[b][month]) * (15-month_day) / 30
    else:
        next_month = (month + 1) % 12
        return temp_mean_mon[b][month] + (temp_mean_mon[b][next_month] - temp_mean_mon[b][month]) * (month_day-15) / 30
    
def get_hum_day(month, month_day, b):
    if month_day < 15:
        last_month = month - 1
        return rela_hum_mean[b][month] + (rela_hum_mean[b][month] - rela_hum_mean[b][last_month]) * (15-month_day) / 30
    else:
        next_month = (month + 1) % 12
        return rela_hum_mean[b][month] + (rela_hum_mean[b][next_month] - rela_hum_mean[b][month]) * (month_day-15) / 30

""" Parameters of buildings """

# daily trend
ele_power_day_change = [[0.5, -0.2, -0.2, -0.2, -0.2, -0.2, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1.5, 1.5, 1],        # programmer
                        [0.5, 1, 1, 2, 1, 0, -0.2, -0.3, -0.3, -0.3, -0.4, -0.3, -0.3, 3, 2, 2, 3, 2, 1, 2, 1, 1, 1, 1],      # wfh at night
                        [2, 2, 2, 2, 2, 2, 2, 2, 3, 3.5, 1.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 2, 3, 3, 2, 2, 2, 2],       # college student
                        [-0.3, -0.3, -0.3, -0.3, -0.3, -0.3, 2, 0, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, 2, 3, 3, 2, 1, -0.2],      # civil servent
                        [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 2, 3, 3, 3, 3, 2, 0, 0, 1, 1]]     # rish second generation

solar_gen_day_change = [[0, 0, 0, 0, 0, 0, -280, -200, -40, 150, 300, 325, 350, 350, 300, 250, 100, -150, -250, -280, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, -140, -100, 0, 120, 210, 230, 249, 240, 103, 98, 12, -123, -140, -150, 0, 0, 0, 0],      # b2
                        [0, 0, 0, 0, 0, 0, -290, -210, -100, 50, 150, 210, 300, 320, 300, 310, 200, 50, -120, -250, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, -300, -300, -300, -250, -150, 0, 129, 150, 92, 0, -170, -300, -300, -300, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, -120, -50, 120, 210, 270, 320, 350, 380, 320, 280, 130, -50, -100, -130, 0, 0, 0, 0]]     # b4

def get_ele_day(d, efficiency, weather_data):      # extra power for ac and heating. efficiency \in [2, 5]
    temp = weather_data[d][0]
    hum = weather_data[d][1]
    if hum > 60:
        hum_ele = (hum - 60) / 20 / efficiency
    if temp > 25:
        temp_ele = temp / 25 / efficiency
    elif temp < 15:
        temp_ele = (30 - temp) / 25 / efficiency
    else:
        temp_ele = 0
    
    return hum_ele + temp_ele

def get_solar_day(d, weather_data, intercept=390, coef=1.18, panels=1):
    temp = weather_data[d][0]
    return (intercept + temp * coef) * panels       # affine

ele_efficienty = [3, 2, 4, 5, 2]
solar_intercept = [350, 368, 320, 290, 400]
solar_coef = [3, 5, 2, 2.5, 5]
solar_panel = [1, 2, 1.5, 0.8, 0.5]

for b in range(5):
    """ weather """
    temp_means = [0.0] * 360     # mean of temperature of each day
    hum_means = [0.0] * 360      # mean of humidity of each day

    temp_day_sigma = 1      # within a month, sigma of the mean of temp
    temp_hour_sigma = 0.01  # within a day, sigma of the temp
    hum_day_sigma = 2
    hum_hour_sigma = 1
    temp_day_change = 4
    hum_day_change = 12

    weather_data = [[0.] * 16 for _ in range(365*24)]
    for d in range(360):
        month = d // 30
        month_day = d % 30
        temp_day_mean = get_temp_day(month, month_day, b)
        hum_day_mean = get_hum_day(month, month_day, b)
        temp_means[d] = temp_day_mean
        hum_means[d] = hum_day_mean
        for j in range(24):
            if j < 12:
                weather_data[d*24+j][0] = random.gauss(temp_day_mean+(temp_day_change/12*j-temp_day_change/2), temp_hour_sigma)
                weather_data[d*24+j][1] = random.gauss(hum_day_mean+(hum_day_change/2-hum_day_change/12*(j-12)), hum_hour_sigma)
            else:
                weather_data[d*24+j][0] = random.gauss(temp_day_mean+(temp_day_change/2-temp_day_change/12*(j-12)), temp_hour_sigma)
                weather_data[d*24+j][1] = random.gauss(hum_day_mean+(hum_day_change/12*j-hum_day_change/2), hum_hour_sigma)
        
    weather_data = np.array(weather_data)

    """ building """
    ele_day_sigma = 0.1
    ele_hour_sigma = 0.01
    solar_day_sigma = 10
    solar_hour_sigma = 5

    building_data = [[0.]*12 for _ in range(365*24)]

    for d in range(365):
        month = d // 30
        ele_mean = get_ele_day(d, ele_efficienty[b], weather_data)
        solar_mean = get_solar_day(d, weather_data, solar_intercept[b], solar_coef[b], solar_panel[b])
        for j in range(24):
            building_data[d*24+j][0] = month+1
            building_data[d*24+j][1] = j if j != 0 else 24
            building_data[d*24+j][7] = random.gauss(ele_mean+ele_power_day_change[b][j], ele_hour_sigma)
            if j < 6 or j > 19:
                building_data[d*24+j][11] = 0
            else:
                building_data[d*24+j][11] = random.gauss(solar_mean+solar_gen_day_change[b][j], solar_hour_sigma)
    building_data = np.array(building_data)
    building_data[building_data<0] = 0

    with open("/home/yunxiang.li/FRL/CityLearn/citylearn/data/my_data/weather_func_{}.csv".format(b+1), "w") as f:
        writer = csv.writer(f)
        writer.writerow(w_header)
        writer.writerows(weather_data)

    with open("/home/yunxiang.li/FRL/CityLearn/citylearn/data/my_data/building_func_{}.csv".format(b+1), "w") as f:
        writer = csv.writer(f)
        writer.writerow(b_header)
        writer.writerows(building_data)
