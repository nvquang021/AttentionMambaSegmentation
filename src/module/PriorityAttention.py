class PCA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=9, groups=dim, padding="same")
        self.prob = nn.Softmax(dim=1)

    def forward(self, x):
        c = reduce(x, 'b c w h -> b c', 'mean')
        x = self.dw(x)
        c_ = reduce(x, 'b c w h -> b c', 'mean')

        # Compute the attention scores
        raise_ch = self.prob(c_ - c)
        att_score = torch.sigmoid(c_ * (1 + raise_ch))
        att_score = att_score.unsqueeze(2).unsqueeze(3)  
        return x * att_score  
class PSA(nn.Module):
    def __init__(self, dim,H=8,W=6):

        super().__init__()
        self.pw = nn.Conv2d(dim, dim, kernel_size=1)
        self.pw_h = nn.Conv2d(H, H, (1, 1))
        self.pw_w = nn.Conv2d(W, W, (1, 1))
        self.prob = nn.Softmax2d()
    def forward(self, x):

        s = reduce(x , 'b c w h -> b w h', 'mean')

        x1 = self.pw(x)
        x_h = self.pw_h(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x_w = self.pw_w(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

        s_ = reduce(x , 'b c w h -> b w h', 'mean')
        s_h = reduce(x_h , 'b c w h -> b w h', 'mean')
        s_w = reduce(x_w , 'b c w h -> b w h', 'mean')

        raise_sp = self.prob(s_ - s)
        raise_h = self.prob(s_h - s)
        raise_w = self.prob(s_w - s)

        att_score = torch.sigmoid(s_*(1 + raise_sp)+s_h*(1 + raise_h)+s_w*(1 + raise_w))
        att_score = att_score.unsqueeze(1)
        return x * att_score