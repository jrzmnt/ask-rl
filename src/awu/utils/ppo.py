import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy


class DropoutMlpExtractor(nn.Module):
    def __init__(self, feature_dim, net_arch, dropout_rate=0.2):
        super().__init__()

        # Policy network with dropout
        policy_layers = []
        last_dim = feature_dim
        for dim in net_arch['pi']:
            policy_layers.extend([
                nn.Linear(last_dim, dim),
                nn.Dropout(p=dropout_rate),
                nn.Tanh()
            ])
            last_dim = dim
        self.policy_net = nn.Sequential(*policy_layers)

        # Value network with dropout
        value_layers = []
        last_dim = feature_dim
        for dim in net_arch['vf']:
            value_layers.extend([
                nn.Linear(last_dim, dim),
                nn.Dropout(p=dropout_rate),
                nn.Tanh()
            ])
            last_dim = dim
        self.value_net = nn.Sequential(*value_layers)

        self.latent_dim_pi = net_arch['pi'][-1]
        self.latent_dim_vf = net_arch['vf'][-1]

    def forward(self, features):
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features):
        return self.policy_net(features)

    def forward_critic(self, features):
        return self.value_net(features)


class DropoutActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, dropout_rate=0.2, **kwargs):
        self.dropout_rate = dropout_rate
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = DropoutMlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            dropout_rate=self.dropout_rate
        )
