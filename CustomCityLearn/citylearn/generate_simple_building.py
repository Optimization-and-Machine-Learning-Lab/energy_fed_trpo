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

""" Parameters of buildings """

# daily trend
ele_power_day_change = [[0.5, -0.2, -0.2, -0.2, -0.2, -0.2, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1.5, 1.5, 1],        # programmer
                        [0.5, 1, 1, 2, 1, 0, -0.2, -0.3, -0.3, -0.3, -0.4, -0.3, -0.3, 3, 2, 2, 3, 2, 1, 2, 1, 1, 1, 1],      # wfh at night
                        [2, 2, 2, 2, 2, 2, 2, 2, 3, 3.5, 1.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 2, 3, 3, 2, 2, 2, 2],       # college student
                        [-0.3, -0.3, -0.3, -0.3, -0.3, -0.3, 2, 0, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, 2, 3, 3, 2, 1, -0.2],      # civil servent
                        [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 2, 3, 3, 3, 3, 2, 0, 0, 1, 1]]     # rich second generation

solar_gen_day_change = [[0, 0, 0, 0, 0, 0, -280, -200, -40, 150, 300, 325, 350, 350, 300, 250, 100, -150, -250, -280, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, -140, -100, 0, 120, 210, 230, 249, 240, 103, 98, 12, -123, -140, -150, 0, 0, 0, 0],      # b2
                        [0, 0, 0, 0, 0, 0, -290, -210, -100, 50, 150, 210, 300, 320, 300, 310, 200, 50, -120, -250, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, -300, -300, -300, -250, -150, 0, 129, 150, 92, 0, -170, -300, -300, -300, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, -120, -50, 120, 210, 270, 320, 350, 380, 320, 280, 130, -50, -100, -130, 0, 0, 0, 0]]     # b4

daily_dem_mean = [np.linspace(0.3, 1.5, 300), np.linspace(0.5, 2, 300), np.linspace(0.8, 3, 300), np.linspace(4, 0.5, 300), np.linspace(5, 0.1, 300)]
daily_solar_mean = [np.linspace(300, 400, 300), np.linspace(700, 1000, 300), np.linspace(300, 500, 300), np.linspace(370, 320, 300), np.linspace(300, 150, 300)]

for b in range(5):
    """ building """
    ele_day_sigma = 0.01
    ele_hour_sigma = 0.01
    solar_day_sigma = 10
    solar_hour_sigma = 5

    building_data = [[0.]*12 for _ in range(300*24)]

    for d in range(300):
        month = d // 30
        ele_mean = daily_dem_mean[b][d]
        solar_mean = daily_solar_mean[b][d]
        for j in range(24):
            building_data[d*24+j][0] = month+1
            building_data[d*24+j][1] = j if j != 0 else 24
            building_data[d*24+j][7] = random.gauss(ele_mean+ele_power_day_change[b][j], ele_hour_sigma)
            if j < 6 or j > 19:
                building_data[d*24+j][11] = 0
            else:
                building_data[d*24+j][11] = random.gauss(solar_mean+solar_gen_day_change[b][j], solar_hour_sigma)
    building_data = np.array(building_data)
    # print(np.where(building_data<0))
    building_data[building_data<0] = 0

    with open("./CityLearn/citylearn/data/my_data/building_simple_{}.csv".format(b+1), "w") as f:
        writer = csv.writer(f)
        writer.writerow(b_header)
        writer.writerows(building_data)
