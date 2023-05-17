# build q table for the simplest env
import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

# env = BanmaEnv()
# env = FLEnv(0)
storage_slices = 81
action_slices = 161

capacity = 6.4   #1.0

storage_options = np.linspace(0, capacity, storage_slices)
# np.arange(storage_slices+1) / storage_slices     # slice the storage to 40 pieces
action_options = np.linspace(-capacity, capacity, action_slices)
print(storage_options)
print()
print(action_options)
print()
storage_unit = storage_options[1]

# exit()

price = np.array([0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.54, 0.54, 0.54, 0.54, 0.54, 0.22, 0.22])
# price = np.array([0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.54, 0.54, 0.54, 0.54, 0.54, 0.22, 0.22])
# building 0
# daily_load = np.array([1.11978309, 0.76816295, 0.87315243, 0.83054701, 0.86020929,
#                         0.95714514, 1.10605024, 0.81951836, 0.88964521, 1.09147795,
#                         1.43532721, 1.38397538, 1.48023339, 1.02653598, 0.97702104,
#                         0.97477498, 1.20762802, 1.56440962, 1.76958789, 2.06710011,
#                         1.86987534, 1.5113112 , 1.28299162, 1.12902468])
# daily_solar = np.array([0.        , 0.        , 0.        , 0.        , 0.        , 0., 
#                             0.03038468, 0.42636759, 1.18112355, 1.8242703 , 2.34805091, 2.63622672,
#                             2.74510293, 2.62572737, 2.32459339, 1.76772156, 1.12567972, 0.56241727,
#                             0.15598965, 0.00660886, 0.        , 0.        , 0.        , 0.        ])
# net_consumption = [ 1.41280447, 0.70136852, 0.71357459, 0.69966889, 0.70094382, 0.71281051,
#   0.6081583 , 1.27543193, 0.64571274,-1.08901961,-1.71112735,-1.79952709,
#  -1.91041168,-1.91681584,-1.68323027,-1.45666642,-0.89256873, 0.07335739,
#   1.45562345, 1.60585605, 2.90677094, 2.38828662, 2.39847597]
# net_consumption = [ 1.54952609, 2.04100792, 2.04791582, 3.03220804, 2.04769488, 1.03764514,
#  -1.6074646 ,-1.86607894,-2.21068284,-2.7305872 ,-3.23236414,-3.11035424,
#  -3.20465527, 0.08429972,-0.39047962,-0.32078849, 1.0231539 , 0.54521353,
#  -0.38122998, 0.66990282, 2.03412813, 2.0444306 , 2.04755359]
# net_consumption = [ 2.15839981, 2.13958229, 2.13772447, 2.15286461, 2.14196697, 2.14013592,
#   1.14998782, 0.86634018, 1.43714935, 1.32453904,-1.09852517,-2.92907478,
#  -3.31848838,-3.40144583,-3.32547382,-3.37746105,-2.95277813,-0.16555454,
#   1.51703665, 2.00583202, 2.14468872, 2.14212241, 2.15302057]
# net_consumption = [ 0.20552437, 0.19691096, 0.20061354, 0.20720042, 0.20602427, 0.21433425,
#   2.49649433, 0.52447044, 0.30098168, 0.30686528,-0.09519197,-0.83959013,
#  -1.47400675,-1.62645962,-1.33998803,-0.87305262,-0.02028693, 0.31899858,
#   2.488313  , 3.50971626, 3.50934611, 2.50371849, 1.51334637]
net_consumption = [ 2.75470902, 1.73420417, 1.74552678, 1.76145477, 1.76126299, 1.74479299,
  1.24304874, 0.97599679, 0.29114834,-0.06587565,-0.3487254 , 0.47586465,
  1.3791062 , 1.22518643, 0.50087719, 1.66314276, 2.27811104, 2.95794071,
  3.18315955, 2.28231462, 0.7474026 , 0.7651242 , 1.74913643]

def value_2_index(value):
    # index = value * (storage_slices - 1)
    index = value / storage_unit
    if abs(index - round(index)) > 0.1:
        print("value", value)
        print("my index", index)
        print("round index", round(index))
        print("bug")
    return round(index)

T = 23
Q = np.zeros((storage_slices,action_slices,T+1))
Q[:,:,T] = 0
Q[:,:,T] = - 2000 * np.reshape(storage_options,[-1,1])        # if the storage at the end is not zero, punish
# print(Q[:, :, -1])

for t in range(T-1,-1,-1):
    # print("t is", t)
    for si, s in enumerate(storage_options):
        # print("si is", si, ", s is", s)
        for ai, a in enumerate(action_options):
            # print("\nai is", ai, ", a is", a)
            # net_con = daily_load[t] - daily_solar[t]
            net_con = net_consumption[t]

            if a > 0.0:                            # charge
                valid_action = min(a, capacity - s)    
            elif a < 0.0 and net_con > 0.0:        # discharge
                valid_action = max(max(a, -net_con), -s)
            else:
                valid_action = 0.0
            if valid_action != a:
                Q[si,ai,t] = - np.inf
                continue
            # print("valid action is", valid_action)
            net_con += valid_action
            net_con = max(net_con, 0)
            # reward = - price[t] * net_con
            reward = -net_con

            sp = s + a
            spi = value_2_index(sp)        # index of storage(state) for the next time step

            if spi >= storage_slices:
                SheSaidWeWillNotGotHere

            # print("si, ai", si, ai, "max next q", np.max(Q[spi,:,t+1]))
            Q[si,ai,t] = reward + np.max(Q[spi,:,t+1])
            # print("spi is", spi)
            # print("next max value", np.max(Q[spi,:,t+1]))
            # print("next max index", np.argmax(Q[spi,:,t+1]))
    # if t == 10:
        # exit()

print("max reward {:.2f}".format(np.max(Q[0,:,0])/24))
# print(np.argmax(Q[0,:,0]))
# print(">"*20)
# print(Q[0, 20, 0])
# print(Q[0, 20, 1])
# print(">"*20)
optimal_action = []
s = 0
ai = np.argmax(Q[0,:,0])
optimal_action.append(action_options[ai])

for t in range(1, T):
    s += action_options[ai]
    si = value_2_index(s)
    ai = np.argmax(Q[si,:,t])
    optimal_action.append(action_options[ai])

optimal_action = list(map(lambda x: round(x, ndigits=2), optimal_action))
print(optimal_action)
# [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.6499999999999999, -0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.75, -0.04999999999999993, -0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]