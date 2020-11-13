import torch
import torchtext
import sacrebleu
def translate_sentence(model, sentence, german, english, device, max_length=50):
    
    if type(sentence) == str:
        split = sentence.split()
        tokens = [german.vocab.stoi[token]  for token in split]
    else:
        tokens = [german.vocab.stoi[token] for token in sentence]
    
    tokens.insert(0, german.vocab.stoi['<sos>'])
    tokens.append(german.vocab.stoi['<eos>'])


    sentence_tensor = torch.LongTensor(tokens).unsqueeze(1).to(device)

    with torch.no_grad():
        hidden, cell = model.encoder(sentence_tensor)

    outputs = [english.vocab.stoi["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        if output.argmax(1).item() == english.vocab.stoi["<eos>"]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]


    return translated_sentence[1:]


def bleu_score(data, model, german, english, device):
  target = []
  model_outputs = []
  for sentence in data:
    ...
