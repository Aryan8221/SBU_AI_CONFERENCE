import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception3D(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(Inception3D, self).__init__()

        self.branch1x1 = nn.Conv3d(in_channels, 64, kernel_size=1)

        self.branch3x3 = nn.Sequential(
            nn.Conv3d(in_channels, 96, kernel_size=1),
            nn.Conv3d(96, 128, kernel_size=3, padding=1)
        )

        self.branch5x5 = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=1),
            nn.Conv3d(16, 32, kernel_size=5, padding=2)
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels, 32, kernel_size=1)
        )

        self.fix_size_conv = nn.Conv3d(256, out_channel, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)

        output = self.fix_size_conv(torch.cat([branch1x1, branch3x3, branch5x5, branch_pool], 1))
        return nn.ReLU(inplace=True)(output)


class DecoderBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock3D, self).__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.upsample(x)
        return self.conv(x)


class InceptionDecoder3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionDecoder3D, self).__init__()

        self.encoder1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.encoder2 = Inception3D(64, 256)
        self.pool2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.encoder3 = Inception3D(256, 512)
        self.pool3 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.encoder4 = Inception3D(512, 512)
        self.pool4 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.decoder4 = DecoderBlock3D(512, 256)
        self.reduce_channels3 = nn.Conv3d(768, 256, kernel_size=1)

        self.decoder3 = DecoderBlock3D(256, 128)
        self.reduce_channels2 = nn.Conv3d(384, 128, kernel_size=1)

        self.decoder2 = DecoderBlock3D(128, 64)
        self.reduce_channels1 = nn.Conv3d(128, 64, kernel_size=1)

        self.decoder1 = DecoderBlock3D(64, out_channels)

        self.upsample_final = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.encoder1(x)
        x1_p = self.pool1(x1)

        x2 = self.encoder2(x1_p)
        x2_p = self.pool2(x2)

        x3 = self.encoder3(x2_p)
        x3_p = self.pool3(x3)

        x4 = self.encoder4(x3_p)
        x4_p = self.pool4(x4)

        d4 = self.decoder4(x4_p)

        d4_concat = torch.cat([d4, x3_p], dim=1)
        d4_concat_reduced = self.reduce_channels3(d4_concat)

        d3 = self.decoder3(d4_concat_reduced)

        d3_concat = torch.cat([d3, x2_p], dim=1)
        d3_concat_reduced = self.reduce_channels2(d3_concat)

        d2 = self.decoder2(d3_concat_reduced)

        d2_concat = torch.cat([d2, x1_p], dim=1)
        d2_concat_reduced = self.reduce_channels1(d2_concat)

        d1 = self.decoder1(d2_concat_reduced)

        final_output = self.upsample_final(d1)

        return final_output


# Example usage
# if __name__ == "__main__":
#     model = InceptionDecoder3D(in_channels=2, out_channels=12)  # 2 input and 12 output channels
#     x = torch.randn((4, 2, 96, 96, 96))  # Example 3D medical data with batch size 4
#     output = model(x)
#     print(output.shape)  # Expect output shape [4, 12, 96, 96, 96]
