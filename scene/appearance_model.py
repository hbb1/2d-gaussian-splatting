import torch
import torch.nn as nn
import torch.nn.functional as F

class UpsampleBlock(nn.Module):
    def __init__(self, num_input_channels, num_output_channels):
        super(UpsampleBlock, self).__init__()
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv = nn.Conv2d(num_input_channels // (2 * 2), num_output_channels, 3, stride=1, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pixel_shuffle(x)
        x = self.conv(x)
        x = self.relu(x)
        return x
    
class AppearanceNetwork(nn.Module):
    def __init__(self, num_input_channels, num_output_channels):
        super(AppearanceNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(num_input_channels, 256, 3, stride=1, padding=1)
        self.up1 = UpsampleBlock(256, 128)
        self.up2 = UpsampleBlock(128, 64)
        self.up3 = UpsampleBlock(64, 32)
        self.up4 = UpsampleBlock(32, 16)
        
        self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, num_output_channels, 3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        # bilinear interpolation
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.sigmoid(x)
        return x

class AppearanceModel:
    def __init__(self, num_embeddings, num_input_channels=67, num_output_channels=3):
        self.appearance_network = AppearanceNetwork(num_input_channels, num_output_channels).cuda()
        
        std = 1e-4
        self._appearance_embeddings = nn.Parameter(torch.empty(num_embeddings, 64).cuda())
        self._appearance_embeddings.data.normal_(0, std)
        
    def get_embedding(self, idx):
        return self._appearance_embeddings[idx]
        
    def training_setup(self, training_args):
        params = [
            {'params': [self._appearance_embeddings], 'lr': training_args.appearance_embeddings_lr, "name": "appearance_embeddings"},
            {'params': self.appearance_network.parameters(), 'lr': training_args.appearance_network_lr, "name": "appearance_network"}
        ]
        self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)
        
    def load_state_dict(self, state_dict):
        self._appearance_embeddings = state_dict["_appearance_embeddings"]
        self.appearance_network.load_state_dict(state_dict["appearance_network"])
        
    def state_dict(self):
        return {
            "_appearance_embeddings": self._appearance_embeddings,
            "appearance_network": self.appearance_network.state_dict()
        }


    
if __name__ == "__main__":
    H, W = 1200//32, 1600//32
    input_channels = 3 + 64
    output_channels = 3
    input = torch.randn(1, input_channels, H, W).cuda()
    model = AppearanceNetwork(input_channels, output_channels).cuda()
    
    output = model(input)
    print(output.shape)