import torch
import torch.nn as nn

class Policy(nn.Module):
    
    def __init__(self, num_inputs, num_outputs, one_hot: bool = False, one_hot_dim: int = 1):

        super(Policy, self).__init__()
        
        self.one_hot = one_hot
        self.one_hot_dim = one_hot_dim
        
        if self.one_hot:

            self.temp_hum_fc_1 = nn.Linear(2, 32)
            self.temp_hum_fc_2 = nn.Linear(32, 32)
            self.other_fc = nn.Linear(num_inputs - one_hot_dim - 2, 32)      # hour, carbon_intensity, non_shiftable_load, electrical_storage_soc, electricity_pricing
            self.one_hot_demand_fc = nn.Linear(self.one_hot_dim, 32)      # demand
            self.one_hot_solar_fc = nn.Linear(self.one_hot_dim, 32)      # solar generation

            self.demand_fc = nn.Linear(64, 32)    # concat temp_hum_2 and one_hot_demand
            self.solar_fc = nn.Linear(64, 32)     # concat temp_hum_2 and one_hot_solar

            self.fc = nn.Linear(96, 64)         # concat demand_fc, solar_fc and other_fc
        
        else:
            self.affine1 = nn.Linear(num_inputs, 64)
            self.affine2 = nn.Linear(64, 256)
            self.affine3 = nn.Linear(256, 64)

        self.action_mean = nn.Linear(64, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.zeros(num_outputs))

        self.saved_actions = []
        self.rewards = []
        self.final_value = 0

    def forward(self, x):

        if self.one_hot:

            if x.dim() == 1:

                x = x.unsqueeze(dim=0)

            # 'hour', 'outdoor_dry_bulb_temperature', 'outdoor_relative_humidity', 'carbon_intensity', 'non_shiftable_load', 'electrical_storage_soc', 'electricity_pricing'

            one_hot_demand = torch.tanh(self.one_hot_demand_fc(x[:, : self.one_hot_dim]))
            one_hot_solar = torch.tanh(self.one_hot_solar_fc(x[:, : self.one_hot_dim]))
            temp_hum = torch.tanh(self.temp_hum_fc_1(x[:, self.one_hot_dim + 2 : self.one_hot_dim + 4]))
            temp_hum = torch.tanh(self.temp_hum_fc_2(temp_hum))
            other = torch.tanh(self.other_fc(torch.hstack((x[:, self.one_hot_dim : self.one_hot_dim + 2], x[ : , -4 : ]))))

            demand = torch.tanh(self.demand_fc(torch.cat((temp_hum, one_hot_demand), dim=1)))
            solar = torch.tanh(self.solar_fc(torch.cat((temp_hum, one_hot_solar), dim=1)))

            x = torch.tanh(self.fc(torch.cat((demand, solar, other), dim=1)))

        else:
            x = torch.tanh(self.affine1(x))
            x = torch.tanh(self.affine2(x))
            x = torch.tanh(self.affine3(x))

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std


class Value(nn.Module):

    def __init__(self, num_inputs, one_hot: bool = False, one_hot_dim: int = 1):
    
        super(Value, self).__init__()
                # state: [onehot, temperature, humidity, battery_storage, consumption, price, hour1, hour2]
    
        self.one_hot = one_hot
        self.one_hot_dim = one_hot_dim
        
        if self.one_hot:

            self.temp_hum_fc_1 = nn.Linear(2, 32)
            self.temp_hum_fc_2 = nn.Linear(32, 32)
            self.other_fc = nn.Linear(num_inputs - one_hot_dim - 2, 32)      # hour, carbon_intensity, non_shiftable_load, electrical_storage_soc, electricity_pricing
            self.one_hot_demand_fc = nn.Linear(self.one_hot_dim, 32)      # demand
            self.one_hot_solar_fc = nn.Linear(self.one_hot_dim, 32)      # solar generation

            self.demand_fc = nn.Linear(64, 32)    # concat temp_hum_2 and one_hot_demand
            self.solar_fc = nn.Linear(64, 32)     # concat temp_hum_2 and one_hot_solar

            self.fc = nn.Linear(96, 64)         # concat demand_fc, solar_fc and other_fc
        
        else:
            self.affine1 = nn.Linear(num_inputs, 64)
            # self.affine2 = nn.Linear(64, 64)
            self.affine2 = nn.Linear(64, 256)
            self.affine3 = nn.Linear(256, 64)

        self.value_head = nn.Linear(64, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):

        if self.one_hot:
            
            if x.dim() == 1:

                x = x.unsqueeze(dim=0)

            # 'hour', 'outdoor_dry_bulb_temperature', 'outdoor_relative_humidity', 'carbon_intensity', 'non_shiftable_load', 'electrical_storage_soc', 'electricity_pricing'

            one_hot_demand = torch.tanh(self.one_hot_demand_fc(x[:, : self.one_hot_dim]))
            one_hot_solar = torch.tanh(self.one_hot_solar_fc(x[:, : self.one_hot_dim]))
            temp_hum = torch.tanh(self.temp_hum_fc_1(x[:, self.one_hot_dim + 2 : self.one_hot_dim + 4]))
            temp_hum = torch.tanh(self.temp_hum_fc_2(temp_hum))
            other = torch.tanh(self.other_fc(torch.hstack((x[:, self.one_hot_dim : self.one_hot_dim + 2], x[ : , -4 : ]))))

            demand = torch.tanh(self.demand_fc(torch.cat((temp_hum, one_hot_demand), dim=1)))
            solar = torch.tanh(self.solar_fc(torch.cat((temp_hum, one_hot_solar), dim=1)))

            x = torch.tanh(self.fc(torch.cat((demand, solar, other), dim=1)))

        else:
            x = torch.tanh(self.affine1(x))
            x = torch.tanh(self.affine2(x))
            x = torch.tanh(self.affine3(x))

        state_values = self.value_head(x)
        return state_values