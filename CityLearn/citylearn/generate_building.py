import csv
import random
import numpy as np

def get_month(day):
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

header = ["Month", "Hour", 
"Day Type", 
"Daylight Savings Status", "Indoor Temperature [C]", "Average Unmet Cooling Setpoint Difference [C]", "Indoor Relative Humidity [%]", 
"Equipment Electric Power [kWh]", 
"DHW Heating [kWh]", "Cooling Load [kWh]", "Heating Load [kWh]", 
"Solar Generation [W/kW]",
]

# ele_power average per month
# ele_power_mean = [2, 2, 1.5, 1, 1, 1, 1, 1, 1, 1.5, 2, 2]
ele_power_mean = [3, 3, 3, 2.5, 1, 1, 1, 1, 1, 2, 3, 4]
# ele_power_mean = [1, 1, 1, 1, 1, 3, 3, 3, 2.5, 1, 0.8, 1]
# ele_power_mean = [4, 4, 4, 4, 3, 2, 2, 2.5, 2.5, 3, 4, 4]
# ele_power_mean = [0.5, 0.5, 0.5, 1, 2, 2, 3, 3, 2, 1, 0.5, 0.5]

# ele_power change in one day w.r.t. mean
# ele_power_day_change = [0.5, -0.2, -0.2, -0.2, -0.2, -0.2, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1.5, 1.5, 1]
ele_power_day_change = [0.5, 1, 1, 2, 1, 0, -0.2, -0.3, -0.3, -0.3, -0.4, -0.3, -0.3, 3, 2, 2, 3, 2, 1, 2, 1, 1, 1, 1]      # b2
# ele_power_day_change = [2, 2, 2, 2, 2, 2, 2, 2, 3, 3.5, 1.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 2, 3, 3, 2, 2, 2, 2]     # b4

# solar_gen_mean = [316, 513, 419, 318, 404, 550, 333, 333, 550, 380, 300, 420]
solar_gen_mean = [319, 400, 434, 478, 523, 598, 449, 421, 412, 451, 434, 378]
# solar_gen_mean = [359, 487, 435, 446, 498, 512, 534, 598, 512, 509, 412, 399]
# solar_gen_mean = [300, 312, 378, 368, 378, 401, 428, 469, 412, 397, 343, 312]
# solar_gen_mean = [545, 587, 512, 451, 409, 398, 343, 397, 408, 499, 539, 498]

# solar_gen change in one day wrt mean
# solar_gen_day_change = [0, 0, 0, 0, 0, 0, -280, -200, -40, 150, 300, 325, 350, 350, 300, 250, 100, -150, -250, -280, 0, 0, 0, 0]
solar_gen_day_change = [0, 0, 0, 0, 0, 0, -140, -100, 0, 120, 210, 230, 249, 240, 103, 98, 12, -123, -140, -150, 0, 0, 0, 0]      # b2
# solar_gen_day_change = [0, 0, 0, 0, 0, 0, 0, 0, -300, -250, -150, 0, 129, 150, 92, 0, -170, -300, 0, 0, 0, 0, 0, 0]     # b4

ele_day_sigma = 0.1
ele_hour_sigma = 0.01
solar_day_sigma = 10
solar_hour_sigma = 5

data = [[0.]*12 for _ in range(365*24)]


with open("./CityLearn/citylearn/data/my_data/building_2.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for i in range(365):
        month = get_month(i+1) - 1
        # ele power mean for today
        ele_mean = random.gauss(ele_power_mean[month], ele_day_sigma)
        # solear mean for today
        solar_mean = random.gauss(solar_gen_mean[month], solar_day_sigma)
        for j in range(24):
            data[i*24+j][0] = month+1
            data[i*24+j][1] = j if j != 0 else 24
            data[i*24+j][7] = random.gauss(ele_mean+ele_power_day_change[j], ele_hour_sigma)
            if j < 6 or j > 19:
                data[i*24+j][11] = 0
            else:
                data[i*24+j][11] = random.gauss(solar_mean+solar_gen_day_change[j], solar_hour_sigma)
    data = np.array(data)
    data[data<0] = 0
    writer.writerows(data)
