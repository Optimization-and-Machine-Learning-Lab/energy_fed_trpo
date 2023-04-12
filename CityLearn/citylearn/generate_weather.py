import csv
import random

def get_month(day):
    day += 1
    if 1 <= day <= 31:
        return 1
    elif 32 <= day <= 59:
        return 2
    elif 60 <= day <= 90:
        return 3
    elif 91 <= day <= 120:
        return 4
    elif 121 <= day <= 151:
        return 5
    elif 152 <= day <= 181:
        return 6
    elif 182 <= day <= 212:
        return 7
    elif 213 <= day <= 243:
        return 8
    elif 244 <= day <= 273:
        return 9
    elif 274 <= day <= 304:
        return 10
    elif 305 <= day <= 334:
        return 11
    elif 335 <= day <= 365:
        return 12

header = ["Outdoor Drybulb Temperature [C]", "Relative Humidity [%]", 
"Diffuse Solar Radiation [W/m2]", "Direct Solar Radiation [W/m2]", 
"6h Prediction Outdoor Drybulb Temperature [C]", "12h Prediction Outdoor Drybulb Temperature [C]", "24h Prediction Outdoor Drybulb Temperature [C]", 
"6h Prediction Relative Humidity [%]", "12h Prediction Relative Humidity [%]", "24h Prediction Relative Humidity [%]", 
"6h Prediction Diffuse Solar Radiation [W/m2]", "12h Prediction Diffuse Solar Radiation [W/m2]", "24h Prediction Diffuse Solar Radiation [W/m2]", 
"6h Prediction Direct Solar Radiation [W/m2]", "12h Prediction Direct Solar Radiation [W/m2]", "24h Prediction Direct Solar Radiation [W/m2]"
]

# outdoor drybulb temp average per month
# Bratislava
# outdoor_dry_temp_mean = [0, 1.9, 6.2, 11.5, 15.9, 19.8, 21.8, 21.5, 16.5, 10.9, 5.6, 1]
# Xi'an
# outdoor_dry_temp_mean = [1, 4, 10, 16, 21, 25, 27, 25, 21, 15, 8, 2]
# HK
# outdoor_dry_temp_mean = [17, 17, 20, 23, 27, 29, 29, 29, 28, 26, 22, 18]
# Reykjavik, Iceland
# outdoor_dry_temp_mean = [0, 0, 0, 3, 6, 9, 11, 7, 4, 3, 2, 1]
# New Zealand
outdoor_dry_temp_mean = [20, 20.5, 19, 16.4, 14.4, 12.1, 11.3, 11.9, 13.3, 14.6, 19.6, 15.65]


# relative humidity
# rela_hum_mean = [77, 70, 58, 49, 49, 50, 49, 51, 52, 62, 76, 80]
# rela_hum_mean = [66, 62, 67, 68, 68, 62, 71, 75, 79, 77, 74, 69]
# rela_hum_mean = [56.2, 69.9, 66.5, 68.2, 69.7, 70, 67.8, 67.5, 63.5, 57, 57.2, 52.8]
# rela_hum_mean = [74.4, 72, 72.6, 69.1, 70, 72, 76.3, 76.8, 75, 75.8, 75.7, 74.6]
rela_hum_mean = [78, 80, 82, 83, 86, 86, 86, 83, 82, 77, 76, 77]

# generate data
temp_day_sigma = 1      # within a month, sigma of the mean of temp
temp_hour_sigma = 0.01  # within a day, sigma of the temp
hum_day_sigma = 2
hum_hour_sigma = 1
data = [[0.] * 16 for _ in range(365*24)]
temp_day_change = 4
hum_day_change = 12


with open("/home/yunxiang.li/FRL/CityLearn/citylearn/data/my_data/weather_5.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for i in range(365):
        month = get_month(i) - 1
        temp_mean = random.gauss(outdoor_dry_temp_mean[month], temp_day_sigma)
        hum_mean = random.gauss(rela_hum_mean[month], hum_day_sigma)
        for j in range(24):
            if j < 12:
                data[i*24+j][0] = random.gauss(temp_mean+(temp_day_change/12*j-temp_day_change/2), temp_hour_sigma)
                data[i*24+j][1] = random.gauss(hum_mean+(hum_day_change/2-hum_day_change/12*(j-12)), hum_hour_sigma)
            else:
                data[i*24+j][0] = random.gauss(temp_mean+(temp_day_change/2-temp_day_change/12*(j-12)), temp_hour_sigma)
                data[i*24+j][1] = random.gauss(hum_mean+(hum_day_change/12*j-hum_day_change/2), hum_hour_sigma)

    writer.writerows(data)

