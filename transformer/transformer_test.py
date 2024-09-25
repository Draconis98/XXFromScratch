import unittest
import torch
from transformer import Transformer, Encoder, Decoder

class TestTransformer(unittest.TestCase):
    def setUp(self):
        self.config = {
            'hidden_size': 128,
            'vocab_size': 1000,
            'max_position_embeddings': 128,
            'layer_norm_eps': 1e-12,
            'num_attention_heads': 16,
            'num_hidden_layers': 4,
            'intermediate_size': 512,
            'hidden_dropout_prob': 0.1
        }
        self.model = Transformer(self.config)

    def test_forward_pass(self):
        batch_size = 2
        seq_length = 10
        input_ids = torch.randint(0, self.config['vocab_size'], (batch_size, seq_length))
        
        output = self.model(input_ids, input_ids)
        
        self.assertEqual(output.size(), (batch_size, seq_length, self.config['vocab_size']))

    def test_encoder_decoder_interaction(self):
        encoder = Encoder(self.config)
        decoder = Decoder(self.config)
        
        input_ids = torch.randint(0, self.config['vocab_size'], (2, 10))
        encoder_output = encoder(input_ids)
        decoder_output = decoder(input_ids, encoder_output)
        
        self.assertEqual(decoder_output.size(), (2, 10, self.config['hidden_size']))

if __name__ == '__main__':
    unittest.main()