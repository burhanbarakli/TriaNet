import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np 

def gauss_kernel_ori(channels=3, cuda=False):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                            [4., 16., 24., 16., 4.],
                            [6., 24., 36., 24., 6.],
                            [4., 16., 24., 16., 4.],
                            [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    if cuda:
        kernel = kernel.cuda()
    return kernel

def gauss_kernel(channels=3, device='cpu'):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    return kernel.to(device)

def downsample(x):
    return x[:, :, ::2, ::2]

def conv_gauss(img, kernel):
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    out = F.conv2d(img, kernel, groups=img.shape[1])
    return out

def upsample(x, channels):
    device = x.device
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
    cc = cc.permute(0, 1, 3, 2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
    x_up = cc.permute(0, 1, 3, 2)
    return conv_gauss(x_up, 4 * gauss_kernel(channels, device=device))

def make_laplace(img, channels):
    device = img.device
    filtered = conv_gauss(img, gauss_kernel(channels, device=device))
    down = downsample(filtered)
    up = upsample(down, channels)
    if up.shape[2] != img.shape[2] or up.shape[3] != img.shape[3]:
        up = nn.functional.interpolate(up, size=(img.shape[2], img.shape[3]))
    diff = img - up
    return diff

def make_laplace_pyramid(img, level, channels):
    device = img.device
    current = img
    pyr = []
    for _ in range(level):
        filtered = conv_gauss(current, gauss_kernel(channels, device=device))
        down = downsample(filtered)
        up = upsample(down, channels)
        if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
            up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
        diff = current - up
        pyr.append(diff)
        current = down
    pyr.append(current)
    return pyr

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
    def forward(self, x):
        avg_out = self.mlp(F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))))
        max_out = self.mlp(F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))))
        channel_att_sum = avg_out + max_out

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.spatial = nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2)
    def forward(self, x):
        x_compress = torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)
        self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out

#Edge-Guided Attention Module
class EGA(nn.Module):
    def __init__(self, in_channels):
        super(EGA, self).__init__()

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 3 , 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid())

        self.cbam = CBAM(in_channels)

    def forward(self, edge_feature, x, pred):
        residual = x
        xsize = x.size()[2:]
        
        pred = torch.sigmoid(pred)
        
        #reverse attention 
        background_att = 1 - pred
        background_x= x * background_att
        
        #boundary attention - edge işareti düzeltmesi
        edge_pred = make_laplace(pred, 1)
        # EGA edge işareti için abs() ekleyerek kararlılığı artırıyoruz
        edge_pred = edge_pred.abs()
        pred_feature = x * edge_pred

        #high-frequency feature
        edge_input = F.interpolate(edge_feature, size=xsize, mode='bilinear', align_corners=True)
        input_feature = x * edge_input

        fusion_feature = torch.cat([background_x, pred_feature, input_feature], dim=1)
        fusion_feature = self.fusion_conv(fusion_feature)

        attention_map = self.attention(fusion_feature)
        fusion_feature = fusion_feature * attention_map

        out = fusion_feature + residual
        out = self.cbam(out)
        return out


class PatternAttention(nn.Module):
    def __init__(self, in_channels):
        super(PatternAttention, self).__init__()

        # YENİ: Gabor'un 4 kanalını tek kanallı bir dikkat haritasına dönüştürecek katman.
        # Bu, modelin 4 yönelimden gelen bilgiyi nasıl birleştireceğini öğrenmesini sağlar.
        num_gabor_filters = 4  # GaborConv'da kullandığımız filtre sayısı
        self.gabor_channel_reducer = nn.Conv2d(in_channels=num_gabor_filters, out_channels=1, kernel_size=1)

        # Gabor ile modüle edilmiş özellik haritasını işlemek için basit bir evrişim bloğu.
        self.pattern_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.cbam = CBAM(in_channels)

    def forward(self, gabor_feature, x):
        xsize = x.size()[2:]

        # Gabor özelliğini ana özellik haritasının boyutuyla eşleştir.
        gabor_input = F.interpolate(gabor_feature, size=xsize, mode='bilinear', align_corners=True)

        # DÜZELTME: 4 kanallı gabor_input'u tek kanallı dikkat haritasına indirge.
        gabor_attention_map = self.gabor_channel_reducer(gabor_input)

        # Dikkat haritasını 0-1 aralığına getirmek için sigmoid uygula.
        gabor_attention_map = torch.sigmoid(gabor_attention_map)

        # Artık boyutlar uyumlu: (B, 256, H, W) * (B, 1, H, W)
        pattern_infused_x = x * gabor_attention_map

        # Bu yeni özelliği basit bir evrişim bloğundan geçir.
        out = self.pattern_conv(pattern_infused_x)

        # Son bir rafinasyon için CBAM kullan.
        out = self.cbam(out)

        return out

# Bu yeni sınıfı kodunuza ekleyin
class GaborConv(nn.Module):
    def __init__(self, in_channels=1, ksize=31, sigma=4.0, lambd=10.0, gamma=0.5, n_filters=4):
        super(GaborConv, self).__init__()
        self.n_filters = n_filters
        kernels = []

        # Farklı yönelimler için Gabor çekirdeklerini (kernel) oluştur
        for theta in np.arange(0, np.pi, np.pi / n_filters):
            # Gabor kernel'ini oluşturan matematiksel formülü PyTorch ile uygula
            # Önce bir koordinat ızgarası oluştur
            grid_size = ksize // 2
            y, x = torch.meshgrid(torch.linspace(-grid_size, grid_size, ksize),
                                  torch.linspace(-grid_size, grid_size, ksize), indexing='ij')

            # Koordinatları döndür
            x_theta = x * torch.cos(torch.tensor(theta)) + y * torch.sin(torch.tensor(theta))
            y_theta = -x * torch.sin(torch.tensor(theta)) + y * torch.cos(torch.tensor(theta))

            # Gabor formülünü uygula
            gb = torch.exp(-.5 * (x_theta ** 2 + gamma ** 2 * y_theta ** 2) / sigma ** 2) * \
                 torch.cos(2 * np.pi * x_theta / lambd)

            kernels.append(gb)

        # Tüm kernelleri tek bir tensörde birleştir
        # Shape: (out_channels, in_channels, K, K) -> (n_filters, 1, ksize, ksize)
        stacked_kernels = torch.stack(kernels).unsqueeze(1)

        # Kernelleri modelin bir parçası yap ama eğitilmesinler (register_buffer)
        self.register_buffer('kernel', stacked_kernels)

    def forward(self, x):
        # F.conv2d ile filtrelemeyi (evrişimi) gerçekleştir
        # padding='same' sayesinde çıktı boyutu girdiyle aynı kalır
        return F.conv2d(x, self.kernel, padding='same')