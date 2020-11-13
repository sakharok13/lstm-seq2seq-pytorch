from torch import nn 
import random
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
      super(Encoder, self).__init__()

      self.hidden_size = hidden_size
      self.num_layers = num_layers

    

      self.embedding = nn.Embedding(input_size, embedding_size)
      self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=0.5)

      self.dropout = nn.Dropout(0.5)
    def forward(self, x):

      embedding = self.dropout(self.embedding(x))

      outputs, (hidden, cell) = self.rnn(embedding)

      
      return hidden, cell



class Decoder(nn.Module):
  def __init__(self, input_size, embedding_size, hidden_size, 
               output_size, num_layers, p=0.5):
    super(Decoder, self).__init__()

    self.hidden_size = hidden_size
    self.num_layers = num_layers

    self.dropout = nn.Dropout(0.5)

    self.embedding = nn.Embedding(input_size, embedding_size)

    self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
    self.fc = nn.Linear(hidden_size, output_size)



  def forward(self, x, hidden, cell):


    x = x.unsqueeze(0)

    embedding = self.dropout(self.embedding(x))


    outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))


    predictions = self.fc(outputs)


    predictions = predictions.squeeze(0)

    return predictions, hidden, cell  


class seq2seq(nn.Module):
  def __init__(self, encoder, decoder):
    super(seq2seq, self).__init__()

    self.encoder = encoder
    self.decoder = decoder

  def forward(self, ger, eng, teacher_force_ration = 0.5):
    batch_size = ger.shape[1]
    target_len = eng.shape[0]
    target_vocab_size = len(english.vocab)

    outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

    hidden, cell = self.encoder(ger)


    x = eng[0]

    for t in range(1, target_len):
      output, hidden, cell = self.decoder(x, hidden, cell)

      outputs[t] = output
      
      best_guess = output.argmax(1)

      x = eng[t] if random.random() < teacher_force_ration else best_guess

    return outputs
