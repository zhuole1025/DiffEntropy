import torch
import torch.nn as nn
import torch.nn.functional as F



#############################################################################
#                 Token Drop Router for K and V Token in DIT                #
#############################################################################

class TDRouter(torch.nn.Module):
    def __init__(self, dim: int, threshold: float = 0.75):
        """
        Initialize the TDRouter layer.

        Args:
            dim (int): The dimension of the input tensor.
            threshold (float): The threshold for the router, determing the ratio of droped tokens.

        Attributes:
            weight (nn.Parameter): Learnable router parameter.

        """
        super().__init__()
        self.threshold = threshold
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        """
        Forward pass through the TDRouter layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying TDRouter.

        """
        B, L, D = x.size() # (batch, token length, feature)
        x_reshaped = x.permute(0, 2, 1).view(-1, L)  # (batch*feature, token length)

        # Generated the pre-masked tensor by learnable router
        _mask = self.fc(x_reshaped) # (batch*feature, token length)
        _mask = _mask.view(B, D, L) # (batch, feature, token length)

        # Get the top-threshold% tokens' position (indices) and generate the mask
        _softmax = F.softmax(_mask, dim=2)
        _, indices = _softmax.topk(int((1-self.threshold)* L), dim=2, largest=True, sorted=True)

        x_reduced = torch.zeros(B, D, int((1-self.threshold)* L)).to(x.device)  # (batch, feature, (1-threshold)*token length)
        x_reduced = x.gather(2, indices)  # (batch, feature, (1-threshold)*token length)

        dropped_x = x_reduced.permute(0, 2, 1)

        return dropped_x

# Insert this code into DIT model forward process
numsteps = 50
routers = torch.nn.ModuleList([
            TDRouter(dim=2000, threshold=0.75) for _ in range(numsteps)
        ])
# router = TDRouter(dim=2000, threshold=0.75)
# input_tensor = torch.randn(4, 2000, 1)
# print(input_tensor.shape)  
# print(input_tensor)
# output_tensor = router(input_tensor)
# print(output_tensor.shape) 
# print(output_tensor)
