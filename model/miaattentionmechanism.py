import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Optional, Tuple


class MiaAttentionHead(nn.Module):
    """
    Implements the attention mechanism described in "Attention Is All You Need", evaluate the attention
    equation computing the dot products of the Q(uery) with all the K(eys) transpose, divide each by 
    sqrt(modelDimension) and apply a softmax function to obtain the weights on the values.

    Parameters:
        modelDimension (int): Dimention of attention
        mask (torch.Tensor): Indices to be masked expresed as a Tensor

    Layer Inputs: query, key, value, mask
    """
    def __init__(self, modelDimension: int):
        super(MiaAttentionHead, self).__init__()
        self.sqrt_moddim = np.sqrt(modelDimension)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:

        # What is happenning here is the computation of the attention formula Q.K(transpose) / Square Root of (ModelDimension)
        # using the provided Q(uery) matrix, transpose of K(ey) and dividing the each item of the result between the square root
        # of the model dimension, the result will be a matrix of attention scores for each word combination.
        attentionScores = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_moddim

        if mask is not None:
            attentionScores.masked_fill_(mask.view(attentionScores.size()), -float('Inf'))

        # Once we have the attention scores we normalize them (this is in order to have attention scores that meets the condition to
        # be in this range 0 <= attentionScores <= 1) because we need a probability based in the dot product that represent
        # how similars are the words (query and keys) combined in pairs
        normalizedAttentionScores = F.softmax(attentionScores, -1)

        # Here we compute the weigthed sum using the normalized attention scores
        # and final result will be the attention matrix represented as context from attention mechanism.
        context = torch.bmm(normalizedAttentionScores, value)

        # Finally we return a tuple with the attention matrix and the normalized attention scores used to compute it
        return context, normalizedAttentionScores

class MiaMultiHeadAttention(nn.Module):
    """
    Implements the Multi-Head Attention sublayer from "Attention Is All You Need", instead to analyze the 
    sequence with one model dimension block (512) we use 'n' heads  doing the same in parallel with a lowest
    dimension of the model, this is model dimension / total heads in this way training is speed up and obtain 
    different representation about how each word relate to each other. 

    Parameters:
        dimensionModel (int): The dimension of K(eys) / V(alues) / Q(ueries). Original Transformer Architecture defines it as 512 by default
        numberOfHeads (int): The number of attention heads this layer will use to analyze relationships between each word in the sequence. 
        Original Transformer Architecture defines it as 8 by default

    Layer Inputs: query, key, value, mask
    """
    def __init__(self, dimensionModel: int = 512, numberOfHeads: int = 8):
        super(MiaMultiHeadAttention, self).__init__()

        # Validates that dimension model and number of heads match, module is 0
        assert dimensionModel % numberOfHeads == 0, "dimensionModel % numberOfHeads must be zero. Values does not match"

        # Computes the dimension for each head. With default values it should be 64 = 512 modelDimension / numberOfHeads 
        self.dimensionHead = int(dimensionModel / numberOfHeads)
        self.numberOfHeads = numberOfHeads

        self.scaledDotAttention = MiaAttentionHead(self.dimensionHead)
        
        self.queryProjection = nn.Linear(dimensionModel, self.dimensionHead * numberOfHeads)
        self.keyProjection = nn.Linear(dimensionModel, self.dimensionHead * numberOfHeads)
        self.valueProjection = nn.Linear(dimensionModel, self.dimensionHead * numberOfHeads)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        batchSize = value.size(0)

        query = self.queryProjection(query).view(batchSize, -1, self.numberOfHeads, self.dimensionHead)  
        key = self.keyProjection(key).view(batchSize, -1, self.numberOfHeads, self.dimensionHead)      
        value = self.valueProjection(value).view(batchSize, -1, self.numberOfHeads, self.dimensionHead)  

        query = query.permute(2, 0, 1, 3).contiguous().view(batchSize * self.numberOfHeads, -1, self.dimensionHead)
        key = key.permute(2, 0, 1, 3).contiguous().view(batchSize * self.numberOfHeads, -1, self.dimensionHead)      
        value = value.permute(2, 0, 1, 3).contiguous().view(batchSize * self.numberOfHeads, -1, self.dimensionHead)  

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.numberOfHeads, 1, 1)

        context, normalizedAttentionScores = self.scaledDotAttention(query, key, value, mask)

        context = context.view(self.numberOfHeads, batchSize, -1, self.dimensionHead)
        context = context.permute(1, 2, 0, 3).contiguous().view(batchSize, -1, self.numberOfHeads * self.dimensionHead)  

        return context, normalizedAttentionScores
