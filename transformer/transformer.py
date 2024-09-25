import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedding(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.embed_dim = config['hidden_size']
        self.vocab_size = config['vocab_size']
        self.max_position_embeddings = config['max_position_embeddings']
        self.layer_norm_eps = config['layer_norm_eps']
        
        self.token_embeddings = nn.Embedding(self.vocab_size, self.embed_dim)
        self.position_embeddings = nn.Embedding(self.max_position_embeddings, self.embed_dim)
        self.layer_norm = nn.LayerNorm(self.embed_dim, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout()
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len).unsqueeze(0)
        input_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        x = input_embeddings + position_embeddings
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x
    

class Attention(nn.Module):
    def __init__(self, embed_dim: int, head_dim: int, mask: bool = False):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, head_dim)
        self.k_proj = nn.Linear(embed_dim, head_dim)
        self.v_proj = nn.Linear(embed_dim, head_dim)
        
        self.mask = mask
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # compute attention scores
        attn_scores = torch.bmm(q, k.transpose(-2, -1))
        # apply causal mask
        if self.mask:
            causal_mask = torch.tril(torch.ones(attn_scores.size(-2), attn_scores.size(-1))).bool()
            attn_scores = attn_scores.masked_fill(causal_mask, -float('inf'))
        # compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        # compute attention output
        attn_output = torch.bmm(attn_weights, v)
        return attn_output
    
    
class CrossAttention(Attention):
    def __init__(self, embed_dim: int, head_dim: int):
        super().__init__(embed_dim, head_dim)
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        q = self.q_proj(x1)
        k = self.k_proj(x2)
        v = self.v_proj(x2)
        
        # compute attention scores
        attn_scores = torch.bmm(q, k.transpose(-2, -1))
        # compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        # compute attention output
        attn_output = torch.bmm(attn_weights, v)
        return attn_output

    
class MultiHeadAttention(nn.Module):
    def __init__(self, config: dict, mask: bool = False):
        super().__init__()
        self.embed_dim = config['hidden_size']
        self.num_heads = config['num_attention_heads']
        self.head_dim = self.embed_dim // self.num_heads
        
        self.heads = nn.ModuleList([Attention(self.embed_dim, self.head_dim, mask=mask) for _ in range(self.num_heads)])
        self.output_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h(hidden_states) for h in self.heads], dim=-1)
        x = self.output_proj(x)
        return x
    

class MaskedMultiHeadAttention(MultiHeadAttention):
    def __init__(self, config: dict):
        super().__init__(config, mask=True)
        
    
class CrossMultiHeadAttention(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.embed_dim = config['hidden_size']
        self.num_heads = config['num_attention_heads']
        self.head_dim = self.embed_dim // self.num_heads
        
        self.heads = nn.ModuleList([CrossAttention(self.embed_dim, self.head_dim) for _ in range(self.num_heads)])
        self.output_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h(hidden_states, encoder_hidden_states) for h in self.heads], dim=-1)
        x = self.output_proj(x)
        return x
    

class MultiHeadCrossAttention(Attention):
    def __init__(self, embed_dim: int, head_dim: int):
        super().__init__(embed_dim, head_dim)
        self.mask = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # compute attention scores
        attn_scores = torch.bmm(q, k.transpose(-2, -1))
        # compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        # compute attention output
        attn_output = torch.bmm(attn_weights, v)
        # project to output space
        output = self.out_proj(attn_output)
        return output
    
class FeedForward(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.embed_dim = config['hidden_size']
        self.intermediate_size = config['intermediate_size']
        
        self.ff_up = nn.Linear(self.embed_dim, self.intermediate_size)
        self.ff_down = nn.Linear(self.intermediate_size, self.embed_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ff_up(x)
        x = self.gelu(x)
        x = self.ff_down(x)
        x = self.dropout(x)
        return x
    
    
class EncoderLayer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.embed_dim = config['hidden_size']
        
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        
    # Pre layer norm
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = self.layer_norm1(x)
        x = x + self.attention(hidden_states)
        hidden_states = self.layer_norm2(x)
        x = x + self.feed_forward(hidden_states)
        return x
    

class Encoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.num_hidden_layers = config['num_hidden_layers']
        
        self.embeddings = Embedding(config)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(self.num_hidden_layers)])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return x
    
    
class DecoderLayer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.embed_dim = config['hidden_size']
        
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.layer_norm3 = nn.LayerNorm(self.embed_dim)
        
        self.masked_attention = MaskedMultiHeadAttention(config)
        self.attention = CrossMultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        
    def forward(self, x: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.layer_norm1(x)
        x = x + self.masked_attention(hidden_states)
        hidden_states = self.layer_norm2(x)
        x = x + self.attention(hidden_states, encoder_hidden_states)
        hidden_states = self.layer_norm3(x)
        x = x + self.feed_forward(hidden_states)
        return x
    
    
class Decoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.num_hidden_layers = config['num_hidden_layers']
        
        self.embeddings = Embedding(config)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(self.num_hidden_layers)])
        
    def forward(self, x: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x, encoder_hidden_states)
        return x
    
    
class Transformer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.embed_dim = config['hidden_size']
        
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.encoder = Encoder(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.decoder = Decoder(config)
        self.lm_head = nn.Linear(self.embed_dim, config['vocab_size'])
        
    def forward(self, x: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        encoder_output = self.encoder(x)
        cross_kv = self.layer_norm1(encoder_output)
        decoder_output = self.decoder(x, cross_kv)
        output = self.layer_norm2(decoder_output)
        output = self.lm_head(output)
        return output