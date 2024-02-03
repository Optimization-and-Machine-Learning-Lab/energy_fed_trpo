import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from CityLearn.citylearn.citylearn import CityLearnEnv

schema_filepath = './CityLearn/citylearn/data/citylearn_challenge_2022_phase_1/schema.json'
env = CityLearnEnv(schema_filepath)

state_record = []   # 1825 * 672
# state_names = ["month", "day_type", "hour", "outdoor_dry_bulb_temperature", "outdoor_dry_bulb_temperature_predicted_6h", "outdoor_dry_bulb_temperature_predicted_12h", "outdoor_dry_bulb_temperature_predicted_24h", "outdoor_relative_humidity", "outdoor_relative_humidity_predicted_6h", "outdoor_relative_humidity_predicted_12h", "outdoor_relative_humidity_predicted_24h", "diffuse_solar_irradiance", "diffuse_solar_irradiance_predicted_6h", "diffuse_solar_irradiance_predicted_12h", "diffuse_solar_irradiance_predicted_24h", "direct_solar_irradiance", "direct_solar_irradiance_predicted_6h", "direct_solar_irradiance_predicted_12h", "direct_solar_irradiance_predicted_24h", "carbon_intensity", "non_shiftable_load", "solar_generation", "electrical_storage_soc", "net_electricity_consumption", "electricity_pricing", "electricity_pricing_predicted_6h", "electricity_pricing_predicted_12h", "electricity_pricing_predicted_24h"]
# 28 states

state = env.reset()
# print(state[0])
# print(len(state[0]))
actions = [[0]] * 5

for i in range(365):
    state = env.reset()
    b_state = state     # 5 * 672
    done = False
    while not done:
        state, _, done, _ = env.step(actions)
        for b in range(5):
            b_state[b].extend(state[b])
    # print(len(b_state))
    # print(len(b_state[0]))
    # break
    state_record.extend(b_state)
    # print(len(state_record))
    # print(len(state_record[0]))
    # if i == 5:
    #     break
state_record = np.array(state_record)

print(len(state_record))

state_record = preprocessing.normalize(state_record)

kmeans = KMeans(n_clusters=6, random_state=0).fit(state_record)
cluster_labels=kmeans.labels_ 
indices = [None] * 5
for b in range(5):
    indices[b] = np.where(cluster_labels==b)[0]
indices = np.array(indices)
indices_b = indices % 5 + 1
indices_day = np.floor((indices / 5.))

print(indices_b)
print(indices_day)
